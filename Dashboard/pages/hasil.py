import streamlit as st
import pandas as pd
from datetime import datetime
from utils import state

def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📊 Hasil Clustering</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Tabel pengelompokan barang, centroid, jarak Euclidean, dan laporan lengkap
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not state.get("cluster_done"):
        st.warning("⚠️ Harap selesaikan **Konfigurasi Clustering** dan jalankan K-Means terlebih dahulu.")
        return

    df_clustered = state.get("df_clustered")
    df_distances = state.get("df_distances")
    model        = state.get("kmeans_model")
    df_agg       = state.get("df_agg")
    sil          = state.get("silhouette_score")
    n_clusters   = state.get("n_clusters")
    label_map    = state.get("cluster_labels")

    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))
    summary = df_full.groupby('Kategori').agg(
        Jumlah_Barang=('Nama Barang', 'count'),
        Rata_Qty=('Qty_2022_2025', 'mean'),
        Min_Qty=('Qty_2022_2025', 'min'),
        Max_Qty=('Qty_2022_2025', 'max'),
    ).reset_index().sort_values('Rata_Qty')

    # ── 1 & 2. Ringkasan Cluster & Centroid (SEJAJAR) ─────────
    col_kiri, col_kanan = st.columns([2, 1])
    
    with col_kiri:
        st.markdown("<div class='section-title'>📌 Ringkasan Cluster</div>", unsafe_allow_html=True)
        colors = {"Laris": "#27ae60", "Sedang": "#f39c12", "Kurang Laris": "#e74c3c"}
        cols = st.columns(len(summary))
        for col, (_, row) in zip(cols, summary.iterrows()):
            c = colors.get(row['Kategori'], "#3498db")
            col.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {c};
                        border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:{c};'>{row['Jumlah_Barang']}</div>
                <div style='font-weight:700; color:{c}; font-size:1rem;'>{row['Kategori']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_kanan:
        st.markdown("<div class='section-title'>🎯 Posisi Centroid</div>", unsafe_allow_html=True)
        centroid_data = []
        df_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
        for cid, _ in df_means.items():
            centroid_val = model.cluster_centers_[cid][0]
            centroid_data.append({"Cluster ID": cid, "Kategori": label_map.get(cid, f"Cluster {cid}"), "Centroid": round(centroid_val, 6)})
        st.table(pd.DataFrame(centroid_data))

    # ── 3 & 4. Tabel Hasil & Rekomendasi (SEJAJAR) ─────────
    st.markdown("---")
    col_tabel, col_rek = st.columns([2, 1])

    with col_tabel:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        all_cats = ["Semua"] + list(df_clustered['Kategori'].unique())
        selected_cat = st.selectbox("Filter Kategori:", all_cats, key="hasil_filter")
        df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
        df_show.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
        if selected_cat != "Semua": df_show = df_show[df_show['Kategori'] == selected_cat]
        st.dataframe(df_show.sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True), use_container_width=True, height=400)
        st.caption(f"Menampilkan {len(df_show):,} barang")

    with col_rek:
        st.markdown("<div class='section-title'>📝 Ringkasan Eksekutif & Strategii</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Laris": ["Prioritaskan ketersediaan stok.", "Jadikan produk sebagai fokus pemasaran.", "Pertahankan kualitas produk dan layanan."],
            "Sedang": ["Pertahankan performa penjualan yang stabil.", "Lakukan promosi secara berkala.", "Pantau perkembangan permintaan pasar."],
            "Kurang Laris": ["Tingkatkan promosi untuk mendorong minat beli.", "Evaluasi strategi pemasaran dan penempatan produk.", "Pantau penjualan secara berkala."],
        }
        for _, row in summary.sort_values('Jumlah_Barang', ascending=False).iterrows():
            kategori = row['Kategori']
            c = colors.get(kategori, "#3498db")
            poin = REKOMENDASI.get(kategori, ["Pantau perkembangan berkala."])
            list_html = "".join([f"<li style='margin-bottom:4px; color:#000000;'>{p}</li>" for p in poin])
            st.markdown(f"""
            <div style='background:#ffffff; border-left:4px solid {c}; border-radius:10px; padding:14px 18px; margin-bottom:12px;'>
                <div style='font-weight:700; color:{c}; font-size:1.05rem;'>{kategori}</div>
                <div style='color:#000000; font-size:0.9rem; margin-top:8px;'>
                    <ul style='margin:6px 0 0 18px; padding:0;'>{list_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 5. Jarak Euclidean ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean ke Centroid</div>", unsafe_allow_html=True)
    st.dataframe(df_distances.head(50), use_container_width=True)

    # ── 6. Download CSV ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>⬇️ Unduh Laporan</div>", unsafe_allow_html=True)
    df_export = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
    df_export.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Hasil Clustering (.csv)",
        data=csv,
        file_name=f"hasil_clustering_toko_asri_mart_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )
