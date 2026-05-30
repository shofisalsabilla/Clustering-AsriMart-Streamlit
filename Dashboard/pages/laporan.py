import streamlit as st
import pandas as pd
from datetime import datetime
from utils import state


def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📋 Laporan Hasil Clustering</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Ringkasan lengkap hasil analisis K-Means Toko Asri Mart
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not state.get("cluster_done"):
        st.warning("⚠️ Harap selesaikan **Konfigurasi Clustering** terlebih dahulu sebelum melihat laporan.")
        return

    df_clustered = state.get("df_clustered")
    df_agg       = state.get("df_agg")
    model        = state.get("kmeans_model")
    sil          = state.get("silhouette_score")
    n_clusters   = state.get("n_clusters")
    label_map    = state.get("cluster_labels")

    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))

    # ── 1. Info Laporan ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>📄 Informasi Laporan</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid #2a4a7f;
                border-radius:12px; padding:16px;'>
        <div style='color:#7f8c8d; font-size:0.8rem;'>Tanggal Dibuat</div>
        <div style='color:white; font-weight:700; font-size:1rem;'>{datetime.now().strftime("%d %B %Y, %H:%M")}</div>
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid #2a4a7f;
                border-radius:12px; padding:16px;'>
        <div style='color:#7f8c8d; font-size:0.8rem;'>Metode</div>
        <div style='color:white; font-weight:700; font-size:1rem;'>K-Means Clustering</div>
    </div>
    """, unsafe_allow_html=True)
    col3.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid #2a4a7f;
                border-radius:12px; padding:16px;'>
        <div style='color:#7f8c8d; font-size:0.8rem;'>Periode Data</div>
        <div style='color:white; font-weight:700; font-size:1rem;'>2022 – 2025</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Ringkasan Eksekutif ───────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Ringkasan Eksekutif</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Barang", f"{len(df_agg):,}")
    col2.metric("Jumlah Cluster", n_clusters)
    col3.metric("Silhouette Score", f"{sil:.4f}")
    col4.metric("Total Qty Keseluruhan", f"{int(df_agg['Qty_2022_2025'].sum()):,}")

    if sil >= 0.7:
        quality = "🟢 Sangat Baik (≥0.70)"
    elif sil >= 0.5:
        quality = "🟡 Baik (0.50–0.69)"
    elif sil >= 0.25:
        quality = "🟠 Cukup (0.25–0.49)"
    else:
        quality = "🔴 Kurang Baik (<0.25)"
    st.markdown(f"**Kualitas Clustering:** {quality}")

    st.markdown("---")

    # ── 3. Ringkasan per Cluster ─────────────────────────────────────
    st.markdown("<div class='section-title'>📌 Ringkasan per Cluster</div>", unsafe_allow_html=True)

    summary = df_full.groupby('Kategori').agg(
        Jumlah_Barang=('Nama Barang', 'count'),
        Total_Qty=('Qty_2022_2025', 'sum'),
        Rata_Qty=('Qty_2022_2025', 'mean'),
        Min_Qty=('Qty_2022_2025', 'min'),
        Max_Qty=('Qty_2022_2025', 'max'),
    ).reset_index().sort_values('Rata_Qty', ascending=False)

    colors = {"Laris": "#27ae60", "Sedang": "#f39c12", "Kurang Laris": "#e74c3c"}
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        c = colors.get(row['Kategori'], "#3498db")
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border:1px solid {c};
                    border-radius:12px; padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{c};'>{row['Jumlah_Barang']}</div>
            <div style='font-weight:700; color:{c}; font-size:1rem;'>{row['Kategori']}</div>
            <div style='color:#7f8c8d; font-size:0.78rem; margin-top:8px;'>
                Total Qty: {int(row['Total_Qty']):,}<br>
                Avg: {row['Rata_Qty']:,.0f}<br>
                Min: {row['Min_Qty']:,.0f} | Max: {row['Max_Qty']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Tabel Ringkasan ───────────────────────────────────────────
    st.markdown("<div class='section-title'>📋 Tabel Ringkasan Cluster</div>", unsafe_allow_html=True)
    summary_display = summary.rename(columns={
        'Kategori': 'Kategori',
        'Jumlah_Barang': 'Jumlah Barang',
        'Total_Qty': 'Total Qty',
        'Rata_Qty': 'Rata-rata Qty',
        'Min_Qty': 'Min Qty',
        'Max_Qty': 'Max Qty',
    })
    st.dataframe(summary_display.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # ── 5. Centroid ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎯 Posisi Centroid</div>", unsafe_allow_html=True)
    centroid_data = []
    df_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
    for cid, _ in df_means.items():
        centroid_val = model.cluster_centers_[cid][0]
        centroid_data.append({
            "Cluster ID": cid,
            "Kategori": label_map.get(cid, f"Cluster {cid}"),
            "Centroid (Normalisasi)": round(centroid_val, 6),
        })
    st.table(pd.DataFrame(centroid_data))

    st.markdown("---")

    # ── 6. Top 10 & Bottom 10 ────────────────────────────────────────
    st.markdown("<div class='section-title'>🏆 10 Barang Terlaris</div>", unsafe_allow_html=True)
    top10 = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].sort_values(
        'Qty_2022_2025', ascending=False).head(10).reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

    st.markdown("<div class='section-title'>📉 10 Barang Paling Kurang Laris</div>", unsafe_allow_html=True)
    bottom10 = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].sort_values(
        'Qty_2022_2025', ascending=True).head(10).reset_index(drop=True)
    bottom10.index += 1
    st.dataframe(bottom10, use_container_width=True)

    st.markdown("---")

    # ── 7. Tabel Lengkap dengan Filter ──────────────────────────────
    st.markdown("<div class='section-title'>📂 Data Lengkap Hasil Clustering</div>", unsafe_allow_html=True)

    all_cats = ["Semua"] + sorted(df_full['Kategori'].unique().tolist())
    selected_cat = st.selectbox("Filter Kategori:", all_cats, key="laporan_filter")

    df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
    df_show.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)

    if selected_cat != "Semua":
        df_show = df_show[df_show['Kategori'] == selected_cat]

    st.dataframe(
        df_show.sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    st.caption(f"Menampilkan {len(df_show):,} dari {len(df_full):,} barang")

    st.markdown("---")

    # ── 8. Download CSV ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>⬇️ Unduh Laporan</div>", unsafe_allow_html=True)

    df_export = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
    df_export.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
    df_export = df_export.sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True)

    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Hasil Clustering (.csv)",
        data=csv,
        file_name=f"hasil_clustering_toko_asri_mart_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

    st.info("✅ Laporan lengkap berhasil dimuat. Gunakan tombol di atas untuk mengunduh data.", icon="ℹ️")
