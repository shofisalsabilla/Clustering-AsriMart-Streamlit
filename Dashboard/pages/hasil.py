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

    # Hapus kolom 'Kategori' bawaan jika ada agar tidak bentrok
    if 'Kategori' in df_clustered.columns:
        df_clustered.drop(columns=['Kategori'], inplace=True)
    if 'Kategori' in df_agg.columns:
        df_agg.drop(columns=['Kategori'], inplace=True)

    # Penggabungan data awal
    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))

    # =========================================================================
    # DETEKSI JUMLAH CLUSTER & DYNAMIC LABELS (SUPPORT K=2 SAMPAI K=5+)
    # =========================================================================
    num_clusters = len(model.cluster_centers_)
    
    # Ambil nilai centroid tiap cluster
    centroids_flat = model.cluster_centers_.flatten()
    
    # Urutkan ID Cluster dari nilai centroid terkecil ke terbesar
    sorted_cluster_ids = np.argsort(centroids_flat)

    # Penentuan skema label dinamis berdasarkan K
    if num_clusters == 2:
        labels = ["Kurang Laris", "Laris"]
    elif num_clusters == 3:
        labels = ["Kurang Laris", "Sedang", "Laris"]
    elif num_clusters == 4:
        labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris"]
    elif num_clusters == 5:
        labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    else:
        # Fallback jika K > 5
        labels = [f"Cluster Rank {i+1}" for i in range(num_clusters)]

    # Buat mapping otomatis ID Cluster -> Label Kategori
    label_map = {}
    centroid_order = []
    custom_cluster_badge = {}
    
    for rank, cid in enumerate(sorted_cluster_ids):
        lbl = labels[rank] if rank < len(labels) else f"Cluster Rank {rank+1}"
        label_map[int(cid)] = lbl
        centroid_order.append((int(cid), lbl))
        custom_cluster_badge[lbl] = int(cid)

    # Urutan filter & rekomendasi (dari tertinggi ke terendah)
    rec_order_list = list(reversed(labels))
    list_pilihan = ["Semua"] + rec_order_list

    # Tetapkan kolom Kategori ke DataFrame utama
    df_full['Kategori'] = df_full['Cluster'].map(label_map).fillna("Lainnya")

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
    # TABEL HASIL PENGELOMPOKAN
    # =========================================================================
    with col_tabel:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        
        df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster']].copy()
        df_show.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
        
        df_show['Kategori'] = df_show['Cluster'].map(label_map).fillna("Lainnya")
        
        selected_cat = st.selectbox("Filter Kategori:", list_pilihan, key="hasil_filter")
        
        if selected_cat != "Semua": 
            df_show = df_show[df_show['Kategori'] == selected_cat]
        
        st.dataframe(
            df_show.sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True), 
            use_container_width=True, 
            height=400
        )
        st.caption(f"Menampilkan {len(df_show):,} barang")

    # =========================================================================
    # REKOMENDASI / STRATEGI
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

        for kategori in rec_order_list:
            if kategori in df_full['Kategori'].values:
                c = get_color(kategori)
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

    # =========================================================================
    # 5. JARAK EUCLIDEAN KE CENTROID
    # =========================================================================
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean ke Centroid</div>", unsafe_allow_html=True)
    
    num_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['Cluster', 'Kategori']]
    X_vals = df_clustered[[num_cols[0]]].values.flatten()

    df_dist_display = pd.DataFrame({'Nama Barang': df_clustered['Nama Barang']})
    
    for cid, label in centroid_order:
        centroid_val = model.cluster_centers_[cid][0]
        dist_to_centroid = np.abs(X_vals - centroid_val)
        df_dist_display[f"Jarak ke Centroid ({label})"] = [f"{v:.4f}" for v in dist_to_centroid]

    st.dataframe(df_dist_display, use_container_width=True)

    st.markdown("---")
    csv = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Clustering (.csv)", data=csv, 
                       file_name=f"hasil_asri_mart_{datetime.now().strftime('%Y%m%d')}.csv", 
                       use_container_width=True, type="primary")
