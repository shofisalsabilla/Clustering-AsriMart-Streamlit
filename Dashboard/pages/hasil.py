import streamlit as st
import pandas as pd
import numpy as np
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

    # Mengambil data dari session state
    df_clustered = state.get("df_clustered").copy()
    model = state.get("kmeans_model")
    df_agg = state.get("df_agg").copy()
    label_map = state.get("cluster_labels")

    # Bersihkan kolom Kategori jika ada di df_agg
    if 'Kategori' in df_agg.columns:
        df_agg.drop(columns=['Kategori'], inplace=True)

    # Penggabungan data
    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))

    # Jika kolom Kategori belum ada, map dari Cluster ID
    if 'Kategori' not in df_full.columns or df_full['Kategori'].isnull().any():
        df_full['Kategori'] = df_full['Cluster'].map(label_map)

    # 1. Ringkasan Cluster & Centroid
    summary = df_full.groupby('Kategori').agg(
        Jumlah_Barang=('Nama Barang', 'count'),
        Rata_Qty=('Qty_2022_2025', 'mean'),
        Min_Qty=('Qty_2022_2025', 'min'),
        Max_Qty=('Qty_2022_2025', 'max'),
    ).reset_index().sort_values('Rata_Qty')

    def get_color(kategori):
        if kategori in ["Sangat Laris", "Laris"]: return "#27ae60"
        if kategori == "Sedang": return "#f39c12"
        return "#e74c3c"

    col_kiri, col_kanan = st.columns([2, 1])
    
    with col_kiri:
        st.markdown("<div class='section-title'>📌 Ringkasan Cluster</div>", unsafe_allow_html=True)
        cols_grid = st.columns(len(summary))
        for col, (_, row) in zip(cols_grid, summary.iterrows()):
            c = get_color(row['Kategori'])
            col.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {c};
                        border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:{c};'>{row['Jumlah_Barang']}</div>
                <div style='font-weight:700; color:{c}; font-size:0.9rem;'>{row['Kategori']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_kanan:
        st.markdown("<div class='section-title'>🎯 Posisi Centroid</div>", unsafe_allow_html=True)
        
        sorted_centroids = sorted(model.cluster_centers_, key=lambda x: x[0])
        centroid_data = []
        for cid, val in enumerate(sorted_centroids):
            lbl = label_map.get(cid, f"Cluster {cid}")
            val_str = ", ".join([f"{v:.8f}" for v in val])
            centroid_data.append({
                "Kategori": lbl,
                "Centroid": f"[{val_str}]"
            })
            
        df_centroid_display = pd.DataFrame(centroid_data)
        st.table(df_centroid_display)

    # 2. Jarak Euclidean ke Centroid
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean ke Centroid</div>", unsafe_allow_html=True)
    
    df_scaled = state.get("df_scaled")
    X_vals = df_scaled[['Qty_2022_2025_Scaled']].values

    df_dist_display = pd.DataFrame({'Nama Barang': df_scaled['Nama Barang']})
    
    sorted_centroids = sorted(model.cluster_centers_, key=lambda x: x[0])
    for cid, centroid_val in enumerate(sorted_centroids):
        lbl = label_map.get(cid, f"Cluster {cid}")
        dists = np.linalg.norm(X_vals - centroid_val, axis=1)
        df_dist_display[f"Jarak ke Centroid ({lbl})"] = [f"{v:.4f}" for v in dists]

    st.dataframe(df_dist_display, use_container_width=True)

    # 3. Tabel Hasil & Rekomendasi
    st.markdown("---")
    col_tabel, col_rek = st.columns([2, 1])

    with col_tabel:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        
        df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
        df_show.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
        
        list_pilihan = ["Semua"] + list(label_map.values())
        selected_cat = st.selectbox("Filter Kategori:", list_pilihan, key="hasil_filter")
        
        if selected_cat != "Semua": 
            df_show = df_show[df_show['Kategori'] == selected_cat]
        
        st.dataframe(
            df_show[['Nama Barang', 'Total Qty (Asli)', 'Kategori', 'Cluster']].sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True), 
            use_container_width=True, 
            height=400
        )
        st.caption(f"Menampilkan {len(df_show):,} barang")

    with col_rek:
        st.markdown("<div class='section-title'>📝 Rekomendasi / Strategi</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Sangat Laris": ["Tingkatkan ketersediaan stok untuk menghindari kehabisan.", "Jadikan produk sebagai produk unggulan.", "Pertahankan strategi pemasaran yang efektif."],
            "Laris": ["Prioritaskan ketersediaan stok.", "Jadikan produk fokus pemasaran.", "Pertahankan kualitas produk dan layanan."],
            "Sedang": ["Pertahankan performa penjualan yang stabil.", "Lakukan promosi secara berkala.", "Pantau perkembangan permintaan pasar."],
            "Kurang Laris": ["Tingkatkan promosi produk.", "Evaluasi strategi pemasaran.", "Pantau penjualan secara berkala."],
            "Sangat Rendah": ["Evaluasi produk dengan tingkat penjualan terendah.", "Pertimbangkan pemberian diskon/promosi.", "Kurangi pengadaan stok."]
        }

        active_categories = list(reversed(list(label_map.values())))
        for kategori in active_categories:
            if kategori in df_full['Kategori'].values:
                c = get_color(kategori)
                poin = REKOMENDASI.get(kategori, ["Pantau perkembangan berkala."])
                list_html = "".join([f"<li style='margin-bottom:4px; color:#000000;'>{p}</li>" for p in poin])
                
                st.markdown(f"""
                <div style='background:#ffffff; border-left:4px solid {c}; border-radius:10px; padding:14px 18px; margin-bottom:12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='font-weight:700; color:{c}; font-size:1rem;'>{kategori}</div>
                    </div>
                    <div style='color:#000000; font-size:0.85rem; margin-top:8px;'>
                        <ul style='margin:6px 0 0 18px; padding:0;'>{list_html}</ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    csv = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Hasil Clustering (.csv)", 
        data=csv, 
        file_name=f"hasil_asri_mart_{datetime.now().strftime('%Y%m%d')}.csv", 
        use_container_width=True, 
        type="primary"
    )
