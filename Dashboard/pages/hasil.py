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

    # Mengambil data dari state
    df_clustered = state.get("df_clustered").copy()
    model        = state.get("kmeans_model")
    df_agg       = state.get("df_agg").copy()

    # Bersihkan kolom Kategori bawaan jika ada
    for df in [df_clustered, df_agg]:
        if 'Kategori' in df.columns:
            df.drop(columns=['Kategori'], inplace=True)

    # Combine data awal
    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))

    # =========================================================================
    # DETEKSI DAN MAPPING OTOMATIS BERDASARKAN HASIL NILAI CENTROID ASLI MODEL
    # =========================================================================
    # Ambil nilai centroid tiap cluster dari model K-Means yang tersimpan
    centroids = model.cluster_centers_.flatten()
    
    # Dapatkan ID cluster diurutkan berdasarkan nilai centroid (rendah ke tinggi)
    sorted_cluster_ids = np.argsort(centroids)
    
    # Map ID cluster asli ke Label Kategori yang sesuai nilainya
    # Id dengan centroid terkecil = Kurang Laris
    # Id dengan centroid sedang = Sedang
    # Id dengan centroid terbesar = Laris
    auto_label_map = {
        int(sorted_cluster_ids[0]): "Kurang Laris",
        int(sorted_cluster_ids[1]): "Sedang",
        int(sorted_cluster_ids[2]): "Laris"
    }

    # Override total kolom Kategori
    df_full['Kategori'] = df_full['Cluster'].map(auto_label_map)
    # =========================================================================

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

    # 1 & 2. Ringkasan Cluster & Centroid
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
        
        centroid_order = [
            (sorted_cluster_ids[0], "Kurang Laris"),
            (sorted_cluster_ids[1], "Sedang"),
            (sorted_cluster_ids[2], "Laris")
        ]
        
        centroid_data = []
        for cid, label in centroid_order:
            val = model.cluster_centers_[cid][0]
            centroid_data.append({
                "Kategori": label,
                "Centroid": f"[{val:.8f}]"
            })
            
        df_centroid_display = pd.DataFrame(centroid_data)
        st.table(df_centroid_display)

    # 3 & 4. Tabel Hasil & Rekomendasi
    st.markdown("---")
    col_tabel, col_rek = st.columns([2, 1])

    # =========================================================================
    # REKOMENDASI / STRATEGI (DIBALIK KHUSUS BAGIAN INI SAJA)
    # =========================================================================
    with col_rek:
        st.markdown("<div class='section-title'>📝 Rekomendasi/Strategi</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Sangat Laris": ["Tingkatkan ketersediaan stok untuk menghindari kehabisan.", "Jadikan produk sebagai produk unggulan.", "Pertahankan strategi pemasaran yang efektif."],
            "Laris": ["Prioritaskan ketersediaan stok.", "Jadikan produk fokus pemasaran.", "Pertahankan kualitas produk dan layanan."],
            "Sedang": ["Pertahankan performa penjualan yang stabil.", "Lakukan promosi secara berkala.", "Pantau perkembangan permintaan pasar."],
            "Kurang Laris": ["Tingkatkan promosi produk.", "Evaluasi strategi pemasaran.", "Pantau penjualan secara berkala."],
            "Sangat Rendah": ["Evaluasi produk dengan tingkat penjualan terendah.", "Pertimbangkan pemberian diskon/promosi.", "Kurangi pengadaan stok."]
        }
        
        # Mapping khusus penanda Cluster ID khusus untuk tampilan kartu Rekomendasi
        custom_cluster_badge = {
            "Laris": 2,        # Diubah jadi Cluster 2
            "Sedang": 1,       # Diubah jadi Cluster 1
            "Kurang Laris": 0  # Tetap Cluster 0
        }

        rec_order_list = ["Laris", "Sedang", "Kurang Laris"]

        for kategori in rec_order_list:
            if kategori in df_full['Kategori'].values:
                c = get_color(kategori)
                # Ambil Cluster ID khusus dari custom_cluster_badge
                cluster_id = custom_cluster_badge.get(kategori, "?")
                poin = REKOMENDASI.get(kategori, ["Pantau perkembangan berkala."])
                list_html = "".join([f"<li style='margin-bottom:4px; color:#000000;'>{p}</li>" for p in poin])
                
                st.markdown(f"""
                <div style='background:#ffffff; border-left:4px solid {c}; border-radius:10px; padding:14px 18px; margin-bottom:12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='font-weight:700; color:{c}; font-size:1rem;'>{kategori}</div>
                        <div style='background:{c}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:bold;'>
                            Cluster {cluster_id}
                        </div>
                    </div>
                    <div style='color:#000000; font-size:0.85rem; margin-top:8px;'>
                        <ul style='margin:6px 0 0 18px; padding:0;'>{list_html}</ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 5. JARAK EUCLIDEAN KE CENTROID
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean ke Centroid</div>", unsafe_allow_html=True)
    
    num_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['Cluster', 'Kategori']]
    X_vals = df_clustered[[num_cols[0]]].values.flatten()

    df_dist_display = pd.DataFrame({'Nama Barang': df_clustered['Nama Barang']})
    
    target_order = [
        (sorted_cluster_ids[0], "Kurang Laris"),
        (sorted_cluster_ids[1], "Sedang"),
        (sorted_cluster_ids[2], "Laris")
    ]
    
    for cid, label in target_order:
        centroid_val = model.cluster_centers_[cid][0]
        dist_to_centroid = np.abs(X_vals - centroid_val)
        df_dist_display[f"Jarak ke Centroid ({label})"] = [f"{v:.4f}" for v in dist_to_centroid]

    st.dataframe(df_dist_display, use_container_width=True)

    st.markdown("---")
    csv = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Clustering (.csv)", data=csv, 
                       file_name=f"hasil_asri_mart_{datetime.now().strftime('%Y%m%d')}.csv", 
                       use_container_width=True, type="primary")
