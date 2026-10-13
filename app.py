"""
Restock & Demand Analyzer Dashboard
=====================================
Aplikasi analitik inventaris interaktif berbasis Streamlit.
Sumber data: inventory.csv & sales_history.csv (folder proyek yang sama)

Jalankan dengan: streamlit run app.py
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Restock & Demand Analyzer",
    page_icon="📦",
    layout="wide",
)

# ─────────────────────────────────────────────
# FUNGSI: MUAT DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Membaca kedua file CSV dan mengembalikannya sebagai DataFrame."""
    inventory = pd.read_csv("inventory.csv")
    sales = pd.read_csv("sales_history.csv", parse_dates=["Date"])
    return inventory, sales


# ─────────────────────────────────────────────
# FUNGSI: HITUNG METRIK INVENTARIS
# ─────────────────────────────────────────────
def compute_inventory_metrics(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan kolom-kolom kalkulasi bisnis ke DataFrame inventaris:

    1. ROP (Reorder Point)
       Rumus: (Daily_Sales_Average × Lead_Time_Days) + Safety_Stock
       ROP menentukan titik stok di mana pemesanan ulang harus dilakukan.

    2. Status Stok
       - 'Kritis'        : Current_Stock < (Daily_Sales_Average × Lead_Time_Days)
                           Stok tidak cukup untuk menutup kebutuhan selama lead time.
       - 'Perlu Restock' : Current_Stock ≤ ROP  (tapi belum kritis)
                           Stok masih ada, namun sudah menyentuh atau lewati titik reorder.
       - 'Stok Berlebih' : Current_Stock > 2.5 × ROP
                           Stok terlalu tinggi, berpotensi membebani modal.
       - 'Stok Aman'     : Semua kondisi lainnya.

    3. Days of Cover
       Rumus: Current_Stock / Daily_Sales_Average
       Jumlah hari stok saat ini dapat memenuhi permintaan.

    4. Rekomendasi Order (unit)
       Hanya dihitung untuk produk berstatus 'Kritis' atau 'Perlu Restock'.
       Target Stok = 2 × ROP
       Rekomendasi Order = Target Stok − Current_Stock
       Nilai negatif dibulatkan ke 0 (tidak perlu pesan).
    """
    df = inventory.copy()

    # --- 1. Reorder Point (ROP) ---
    df["ROP"] = (df["Daily_Sales_Average"] * df["Lead_Time_Days"]) + df["Safety_Stock"]

    # --- 2. Status Stok ---
    demand_during_lead_time = df["Daily_Sales_Average"] * df["Lead_Time_Days"]

    conditions = [
        df["Current_Stock"] < demand_during_lead_time,                    # Kritis
        df["Current_Stock"] <= df["ROP"],                                  # Perlu Restock
        df["Current_Stock"] > 2.5 * df["ROP"],                            # Stok Berlebih
    ]
    choices = ["Kritis", "Perlu Restock", "Stok Berlebih"]
    # Gunakan numpy.select untuk memetakan kondisi ke label status
    import numpy as np
    df["Status_Stok"] = np.select(conditions, choices, default="Stok Aman")

    # --- 3. Days of Cover ---
    # Hindari pembagian dengan nol; produk tanpa penjualan diberi nilai sangat besar
    df["Days_of_Cover"] = df.apply(
        lambda r: round(r["Current_Stock"] / r["Daily_Sales_Average"], 1)
        if r["Daily_Sales_Average"] > 0
        else 9999,
        axis=1,
    )

    # --- 5. Rekomendasi Order ---
    # Target Stok = 2 × ROP (standar baku industri untuk buffer ganda)
    df["Target_Stok"] = 2 * df["ROP"]
    df["Rekomendasi_Order"] = df.apply(
        lambda r: max(0, round(r["Target_Stok"] - r["Current_Stock"]))
        if r["Status_Stok"] in ("Kritis", "Perlu Restock")
        else 0,
        axis=1,
    )

    # --- 6. Nilai Aset Inventaris ---
    df["Nilai_Inventaris"] = df["Current_Stock"] * df["Unit_Cost"]

    return df


# ─────────────────────────────────────────────
# FUNGSI: ANALISIS TREN PENJUALAN
# ─────────────────────────────────────────────
def analyze_sales_trend(sales_item: pd.DataFrame) -> dict:
    """
    Menganalisis pola permintaan dari riwayat penjualan satu produk.

    Metode:
    - Tren  : Bandingkan rata-rata penjualan 30 hari pertama vs 30 hari terakhir.
              Selisih > 10% = Tren Naik/Turun; sisanya = Stabil.
    - Lonjakan mendadak : Hari dengan penjualan > 3× rata-rata harian keseluruhan.
    - Hari tanpa penjualan: Hari yang tidak tercatat dalam data (dianggap 0 unit terjual).
    """
    if sales_item.empty:
        return {"trend_label": "Data Tidak Cukup", "spikes": 0, "zero_days": 0, "avg_daily": 0}

    # Lengkapi tanggal (isi hari tanpa transaksi dengan 0)
    full_range = pd.date_range(sales_item["Date"].min(), sales_item["Date"].max(), freq="D")
    daily = (
        sales_item.groupby("Date")["Units_Sold"]
        .sum()
        .reindex(full_range, fill_value=0)
        .reset_index()
    )
    daily.columns = ["Date", "Units_Sold"]

    avg_daily = daily["Units_Sold"].mean()

    # Tren: 30 hari pertama vs 30 hari terakhir
    first_30 = daily.head(30)["Units_Sold"].mean() if len(daily) >= 30 else daily["Units_Sold"].mean()
    last_30  = daily.tail(30)["Units_Sold"].mean() if len(daily) >= 30 else daily["Units_Sold"].mean()

    if first_30 == 0:
        trend_label = "Stabil"
    else:
        change_pct = (last_30 - first_30) / first_30 * 100
        if change_pct > 10:
            trend_label = "📈 Tren Naik"
        elif change_pct < -10:
            trend_label = "📉 Tren Turun"
        else:
            trend_label = "➡️ Stabil"

    # Lonjakan mendadak: transaksi > 3× rata-rata harian
    spike_threshold = 3 * avg_daily
    spikes = int((daily["Units_Sold"] > spike_threshold).sum()) if avg_daily > 0 else 0

    # Hari tanpa penjualan
    zero_days = int((daily["Units_Sold"] == 0).sum())

    return {
        "trend_label": trend_label,
        "spikes": spikes,
        "zero_days": zero_days,
        "avg_daily": round(avg_daily, 2),
        "daily_df": daily,
        "first_30_avg": round(first_30, 2),
        "last_30_avg": round(last_30, 2),
    }


# ─────────────────────────────────────────────
# HELPER: WARNA STATUS STOK
# ─────────────────────────────────────────────

# Palet latar belakang (pastel) dan teks (gelap/terang) untuk setiap status.
# Warna latar dipilih cukup terang agar teks tetap terbaca di semua layar.
STATUS_BG_COLOR = {
    "Kritis":        "#fecaca",  # merah pastel
    "Perlu Restock": "#fed7aa",  # oranye pastel
    "Stok Berlebih": "#bfdbfe",  # biru pastel
    "Stok Aman":     "#bbf7d0",  # hijau pastel
}
STATUS_TEXT_COLOR = {
    "Kritis":        "#7f1d1d",  # merah tua — kontras tinggi di atas merah pastel
    "Perlu Restock": "#7c2d12",  # coklat tua — kontras di atas oranye pastel
    "Stok Berlebih": "#1e3a5f",  # biru tua  — kontras di atas biru pastel
    "Stok Aman":     "#14532d",  # hijau tua — kontras di atas hijau pastel
}
STATUS_COLOR = {
    "Kritis":        "#ef4444",  # merah (dipakai di pie chart & legenda)
    "Perlu Restock": "#f97316",  # oranye
    "Stok Berlebih": "#3b82f6",  # biru
    "Stok Aman":     "#22c55e",  # hijau
}


def highlight_row(row):
    """
    Mewarnai seluruh baris berdasarkan nilai kolom 'Status Stok'.
    Mengembalikan list style CSS satu string per kolom — pola standar
    pandas Styler.apply(axis=1).
    """
    status = row.get("Status Stok", "")
    bg    = STATUS_BG_COLOR.get(status, "#ffffff")
    color = STATUS_TEXT_COLOR.get(status, "#1f2328")
    base  = f"background-color: {bg}; color: {color};"
    # Kolom Status Stok diberi tambahan font-weight bold agar menonjol
    styles = []
    for col in row.index:
        if col == "Status Stok":
            styles.append(base + " font-weight: 700;")
        else:
            styles.append(base)
    return styles


def color_status(val):
    """Dipakai untuk tabel ringkasan distribusi (satu sel)."""
    color = STATUS_COLOR.get(val, "#6b7280")
    return f"color: {color}; font-weight: 600;"


# ─────────────────────────────────────────────
# MAIN: APLIKASI.
# ─────────────────────────────────────────────
def main():
    # Muat data
    inventory_raw, sales_raw = load_data()
    df_inv = compute_inventory_metrics(inventory_raw)

    # ── HEADER ────────────────────────────────
    st.title("📦 Restock & Demand Analyzer")
    st.caption("Dashboard analitik inventaris real-time berbasis data CSV lokal.")
    st.divider()

    # ── FILTER KATEGORI (halaman utama) ───────
    all_categories = sorted(df_inv["Product_Category"].unique().tolist())
    selected_cats = st.multiselect(
        "🔽 Filter Kategori Produk",
        options=all_categories,
        default=all_categories,
        help="Pilih satu atau lebih kategori untuk memfilter tabel dan grafik.",
    )
    # Jika tidak ada yang dipilih, tampilkan semua
    if selected_cats:
        df_filtered = df_inv[df_inv["Product_Category"].isin(selected_cats)].copy()
        sales_filtered = sales_raw[sales_raw["Item_ID"].isin(df_filtered["Item_ID"])].copy()
    else:
        df_filtered = df_inv.copy()
        sales_filtered = sales_raw.copy()

    st.caption(f"Menampilkan **{len(df_filtered)}** dari **{len(df_inv)}** produk.")
    st.divider()

    # ─────────────────────────────────────────
    # SEKSI 1: KPI CARDS
    # ─────────────────────────────────────────
    total_asset_value = df_filtered["Nilai_Inventaris"].sum()
    critical_count    = int((df_filtered["Status_Stok"] == "Kritis").sum())

    # Produk penjualan tertinggi: berdasarkan Daily_Sales_Average (dari data inventaris)
    # — menggunakan Daily_Sales_Average dari inventory.csv sebagai representasi demand rate
    if not df_filtered.empty:
        top_product_row = df_filtered.loc[df_filtered["Daily_Sales_Average"].idxmax()]
        top_product_name = top_product_row["Product_Name"]
        top_product_avg  = top_product_row["Daily_Sales_Average"]
    else:
        top_product_name = "-"
        top_product_avg  = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Total Nilai Aset Inventaris",
            value=f"Rp {total_asset_value:,.0f}",
            help="Current_Stock × Unit_Cost untuk semua produk yang difilter.",
        )
    with col2:
        st.metric(
            label="🚨 Produk Stok Kritis",
            value=f"{critical_count} Produk",
            delta=f"-{critical_count}" if critical_count > 0 else "Aman",
            delta_color="inverse",
            help="Produk dengan stok di bawah kebutuhan selama lead time.",
        )
    with col3:
        st.metric(
            label="🏆 Produk Penjualan Tertinggi",
            value=top_product_name,
            delta=f"~{top_product_avg} unit/hari",
            help="Produk dengan rata-rata penjualan harian tertinggi.",
        )

    st.divider()

    # ─────────────────────────────────────────
    # SEKSI 2: TABEL ANALISIS INVENTARIS
    # ─────────────────────────────────────────
    st.subheader("📋 Tabel Analisis Inventaris & Rekomendasi Restock")

    # Pilih dan rename kolom untuk tampilan
    display_cols = {
        "Product_Name":       "Nama Produk",
        "Current_Stock":      "Stok Saat Ini",
        "Daily_Sales_Average":"Rata² Penjualan/Hari",
        "Lead_Time_Days":     "Lead Time (Hari)",
        "ROP":                "ROP",
        "Status_Stok":        "Status Stok",
        "Days_of_Cover":      "Days of Cover",
        "Rekomendasi_Order":  "Rekomendasi Order (Unit)",
    }
    df_display = df_filtered[list(display_cols.keys())].rename(columns=display_cols)

    # Terapkan warna latar belakang seluruh baris + teks kontras berdasarkan Status Stok
    styled = (
        df_display.style
        .apply(highlight_row, axis=1)          # warna latar + teks seluruh baris
        .format({
            "Rata² Penjualan/Hari": "{:.2f}",
            "ROP":                  "{:.1f}",
            "Days of Cover":        "{:.1f}",
        })
        .set_properties(**{"font-size": "13px"})
    )

    # Manajemen lebar kolom agar proporsional dan tidak ada teks terpotong
    column_config = {
        "Nama Produk":              st.column_config.TextColumn("Nama Produk",          width="medium"),
        "Stok Saat Ini":            st.column_config.NumberColumn("Stok Saat Ini",       width="small"),
        "Rata² Penjualan/Hari":     st.column_config.NumberColumn("Rata²/Hari",          width="small",  format="%.2f"),
        "Lead Time (Hari)":         st.column_config.NumberColumn("Lead Time",           width="small"),
        "ROP":                      st.column_config.NumberColumn("ROP",                 width="small",  format="%.1f"),
        "Status Stok":              st.column_config.TextColumn("Status",           width="small"),
        "Days of Cover":            st.column_config.NumberColumn("Days of Cover",       width="small",  format="%.1f"),
        "Rekomendasi Order (Unit)": st.column_config.NumberColumn("Rekomendasi Order",  width="small"),
    }

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )

    # Legenda status
    legend_cols = st.columns(4)
    for i, (status, color) in enumerate(STATUS_COLOR.items()):
        legend_cols[i].markdown(
            f"<span style='color:{color}; font-weight:600;'>■</span> {status}",
            unsafe_allow_html=True,
        )

    st.divider()

    # ─────────────────────────────────────────
    # SEKSI 3: GRAFIK TREN PENJUALAN
    # ─────────────────────────────────────────
    st.subheader("📈 Tren Penjualan Harian & Analisis Pola Permintaan")

    # Pilih produk untuk grafik
    product_options = df_filtered["Product_Name"].tolist()
    if not product_options:
        st.warning("Tidak ada produk yang sesuai dengan filter.")
        return

    selected_product = st.selectbox(
        "Pilih produk untuk analisis tren:",
        options=product_options,
        index=0,
    )

    # Ambil Item_ID dari produk terpilih
    selected_id = df_filtered.loc[
        df_filtered["Product_Name"] == selected_product, "Item_ID"
    ].values[0]

    # Filter data penjualan untuk produk terpilih
    sales_item = sales_filtered[sales_filtered["Item_ID"] == selected_id].copy()

    trend_info = analyze_sales_trend(sales_item)

    # ── Panel metrik tren ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tren Permintaan",    trend_info["trend_label"])
    m2.metric("Rata-rata/Hari",     f"{trend_info['avg_daily']} unit")
    m3.metric("Lonjakan Mendadak",  f"{trend_info['spikes']} hari",
              help="Hari dengan penjualan > 3× rata-rata harian")
    m4.metric("Hari Tanpa Penjualan", f"{trend_info['zero_days']} hari",
              help="Hari dalam rentang data tanpa transaksi tercatat")

    # ── Grafik ──
    if "daily_df" in trend_info and not trend_info["daily_df"].empty:
        daily_df = trend_info["daily_df"]
        avg_line = trend_info["avg_daily"]

        fig = go.Figure()

        # Garis penjualan harian
        fig.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Units_Sold"],
            mode="lines+markers",
            name="Penjualan Harian",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
        ))

        # Garis rata-rata
        fig.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=[avg_line] * len(daily_df),
            mode="lines",
            name=f"Rata-rata ({avg_line} unit)",
            line=dict(color="#f97316", width=1.5, dash="dash"),
        ))

        # Garis ambang lonjakan (3× rata-rata)
        if avg_line > 0:
            spike_threshold = 3 * avg_line
            fig.add_trace(go.Scatter(
                x=daily_df["Date"],
                y=[spike_threshold] * len(daily_df),
                mode="lines",
                name=f"Ambang Lonjakan (3× = {spike_threshold:.1f})",
                line=dict(color="#ef4444", width=1, dash="dot"),
            ))

            # Tandai titik lonjakan
            spikes_df = daily_df[daily_df["Units_Sold"] > spike_threshold]
            if not spikes_df.empty:
                fig.add_trace(go.Scatter(
                    x=spikes_df["Date"],
                    y=spikes_df["Units_Sold"],
                    mode="markers",
                    name="Lonjakan",
                    marker=dict(color="#ef4444", size=10, symbol="star"),
                ))

        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Unit Terjual",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#f7f8fa",
            paper_bgcolor="#ffffff",
            hovermode="x unified",
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
        fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

        st.plotly_chart(fig, use_container_width=True)

        # ── Ringkasan tren otomatis ──
        with st.expander("🔍 Detail Analisis Pola Permintaan", expanded=False):
            st.markdown(f"""
| Metrik | Nilai |
|---|---|
| Rata-rata 30 Hari Pertama | **{trend_info['first_30_avg']} unit/hari** |
| Rata-rata 30 Hari Terakhir | **{trend_info['last_30_avg']} unit/hari** |
| Label Tren Otomatis | **{trend_info['trend_label']}** |
| Jumlah Lonjakan Mendadak (>3× avg) | **{trend_info['spikes']} hari** |
| Hari Tanpa Penjualan | **{trend_info['zero_days']} hari** |
            """)
            st.caption(
                "Tren ditentukan berdasarkan perubahan persentase rata-rata 30 hari pertama "
                "vs 30 hari terakhir: >10% = Naik, <-10% = Turun, sisanya = Stabil."
            )
    else:
        st.info("Data penjualan untuk produk ini tidak tersedia dalam rentang filter.")

    st.divider()

    # ─────────────────────────────────────────
    # SEKSI 4: RINGKASAN STATUS STOK
    # ─────────────────────────────────────────
    st.subheader("📊 Distribusi Status Stok")

    status_counts = df_filtered["Status_Stok"].value_counts().reset_index()
    status_counts.columns = ["Status", "Jumlah Produk"]

    col_chart, col_table = st.columns([1.5, 1])

    with col_chart:
        color_map = {
            "Kritis":        "#ef4444",
            "Perlu Restock": "#f97316",
            "Stok Berlebih": "#3b82f6",
            "Stok Aman":     "#22c55e",
        }
        fig_pie = px.pie(
            status_counts,
            names="Status",
            values="Jumlah Produk",
            color="Status",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label", pull=[0.04] * len(status_counts))
        fig_pie.update_layout(
            showlegend=True,
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.dataframe(
            status_counts.style.applymap(color_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    # ─────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────
    st.divider()
    st.caption("© 2025 Restock & Demand Analyzer · Dibuat dengan Streamlit & Plotly · Data dari inventory.csv & sales_history.csv")


if __name__ == "__main__":
    main()
