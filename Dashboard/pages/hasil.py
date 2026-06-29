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

    # ── 1 & 2. Ringkasan Cluster & Centroid (DIBUAT SEJAJAR) ─────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='section-title'>📌 Ringkasan Cluster</div>", unsafe_allow_html=True)
        colors = {"Laris": "#27ae60", "Sedang": "#f39c12", "Kurang Laris": "#e74c3c"}
        cols = st.columns(len(summary))
        for col, (_, row) in zip(cols, summary.iterrows()):
            c = colors.get(row['Kategori'], "#3498db")
            col.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {c}; border-radius:12px; padding:10px; text-align:center;'>
                <div style='font-size:1.5rem; font-weight:800; color:{c};'>{row['Jumlah_Barang']}</div>
                <div style='font-weight:700; color:{c}; font-size:0.9rem;'>{row['Kategori']}</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>🎯 Posisi Centroid</div>", unsafe_allow_html=True)
        centroid_data = []
        df_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
        for cid, _ in df_means.items():
            centroid_data.append({"Kategori": label_map.get(cid, f"Cluster {cid}"), "Centroid": round(model.cluster_centers_[cid][0], 4)})
        st.table(pd.DataFrame(centroid_data))

    # ── 3 & 4. Tabel Hasil & Ringkasan Eksekutif (DIBUAT SEJAJAR) ─────
    st.markdown("---")
    col_tabel, col_rekomendasi = st.columns([2, 1])

    with col_tabel:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        all_cats = ["Semua"] + list(df_clustered['Kategori'].unique())
        selected_cat = st.selectbox("Filter Kategori:", all_cats, key="hasil_filter")
        df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
        df_show.rename(columns={'Qty_2022_2025': 'Total Qty'}, inplace=True)
        if selected_cat != "Semua": df_show = df_show[df_show['Kategori'] == selected_cat]
        st.dataframe(df_show.sort_values('Total Qty', ascending=False), use_container_width=True, height=300)

    with col_rekomendasi:
        st.markdown("<div class='section-title'>📝 Rekomendasi</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Laris": ["Prioritaskan stok.", "Fokus pemasaran."],
            "Sedang": ["Promosi berkala.", "Pantau permintaan."],
            "Kurang Laris": ["Tingkatkan promosi.", "Evaluasi produk."]
        }
        for _, row in summary.iterrows():
            kategori = row['Kategori']
            c = colors.get(kategori, "#3498db")
            poin = REKOMENDASI.get(kategori, ["Pantau berkala."])
            st.markdown(f"""
            <div style='background:#ffffff; border-left:4px solid {c}; border-radius:8px; padding:10px; margin-bottom:8px;'>
                <b style='color:#000;'>{kategori}</b>
                <ul style='color:#000; margin-left:-15px; font-size:0.8rem;'>{''.join([f"<li>{p}</li>" for p in poin])}</ul>
            </div>""", unsafe_allow_html=True)

    # ── 5 & 6. Jarak & Download ──────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean & Unduh</div>", unsafe_allow_html=True)
    st.dataframe(df_distances.head(10), use_container_width=True)
    
    if st.download_button("⬇️ Download CSV Hasil", data=df_full.to_csv(index=False).encode('utf-8'), 
                          file_name="hasil_clustering.csv", mime="text/csv", use_container_width=True, type="primary"):
        pass
