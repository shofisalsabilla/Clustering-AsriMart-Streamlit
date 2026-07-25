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
    scaler       = state.get("scaler") 

    # Hapus kolom 'Kategori' bawaan jika ada agar tidak bentrok
    if 'Kategori' in df_clustered.columns:
        df_clustered.drop(columns=['Kategori'], inplace=True)
    if 'Kategori' in df_agg.columns:
        df_agg.drop(columns=['Kategori'], inplace=True)

    # Penggabungan data awal
    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))

    # =========================================================================
    # LOGIKA PEMETAAN LABEL BERDASARKAN RATA-RATA TOTAL QTY ASLI KELOMPOK
    # =========================================================================
    # Hitung rata-rata penjualan asli untuk setiap Cluster ID
    cluster_means = df_full.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
    sorted_cluster_ids = cluster_means.index.tolist()  # Index cluster dari Rata-rata Qty Terendah ke Tertinggi

    # Peta penamaan berdasarkan urutan penjualan asli
    if len(sorted_cluster_ids) == 3:
        corrected_label_map = {
            sorted_cluster_ids[0]: "Kurang Laris", # Qty Terendah
            sorted_cluster_ids[1]: "Sedang",       # Qty Menengah
            sorted_cluster_ids[2]: "Laris"         # Qty Tertinggi
        }
    else:
        custom_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
        corrected_label_map = {
            cid: custom_labels[i] if i < len(custom_labels) else f"Cluster {cid}"
            for i, cid in enumerate(sorted_cluster_ids)
        }

    # Tetapkan kolom Kategori yang valid ke DataFrame utama
    df_full['Kategori'] = df_full['Cluster'].map(corrected_label_map)
    # =========================================================================

    summary = df_full.groupby('Kategori').agg(
        Jumlah_Barang=('Nama Barang', 'count'),
        Rata_Qty=('Qty_2022_2025', 'mean'),
        Min_Qty=('Qty_2022_2025', 'min'),
        Max_Qty=('Qty_2022_2025', 'max'),
    ).reset_index().sort_values('Rata_Qty')

    # Warna dinamis berdasarkan kategori
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
        inv_label_map = {v: k for k, v in corrected_label_map.items()}
        centroid_data = []
        
        # Mengambil data centroid tiap kategori
        for _, row in summary.iterrows():
            kategori = row['Kategori']
            cid = inv_label_map.get(kategori)
            if cid is not None:
                centroid_val = model.cluster_centers_[cid][0]
                centroid_data.append({
                    "Kategori": kategori, 
                    "Centroid": f"[{centroid_val:.8f}]"
                })
        
        # Urutkan tampilan tabel Centroid: Kurang Laris -> Sedang -> Laris
        df_centroid_display = pd.DataFrame(centroid_data)
        custom_order = ["Kurang Laris", "Sedang", "Laris"]
        df_centroid_display['Kategori'] = pd.Categorical(df_centroid_display['Kategori'], categories=custom_order, ordered=True)
        df_centroid_display = df_centroid_display.sort_values('Kategori').reset_index(drop=True)

        st.table(df_centroid_display)

    # 3 & 4. Tabel Hasil & Rekomendasi
    st.markdown("---")
    col_tabel, col_rek = st.columns([2, 1])

    with col_tabel:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        all_cats = ["Semua"] + list(df_full['Kategori'].unique())
        selected_cat = st.selectbox("Filter Kategori:", all_cats, key="hasil_filter")
        df_show = df_full[['Nama Barang', 'Qty_2022_2025', 'Cluster', 'Kategori']].copy()
        df_show.rename(columns={'Qty_2022_2025': 'Total Qty (Asli)'}, inplace=True)
        if selected_cat != "Semua": 
            df_show = df_show[df_show['Kategori'] == selected_cat]
        
        st.dataframe(df_show.sort_values('Total Qty (Asli)', ascending=False).reset_index(drop=True), use_container_width=True, height=400)
        st.caption(f"Menampilkan {len(df_show):,} barang")

    with col_rek:
        st.markdown("<div class='section-title'>📝 Rekomendasi/Strategi</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Sangat Laris": ["Tingkatkan ketersediaan stok untuk menghindari kehabisan.", "Jadikan produk sebagai produk unggulan.", "Pertahankan strategi pemasaran yang efektif."],
            "Laris": ["Prioritaskan ketersediaan stok.", "Jadikan produk fokus pemasaran.", "Pertahankan kualitas produk dan layanan."],
            "Sedang": ["Pertahankan performa penjualan yang stabil.", "Lakukan promosi secara berkala.", "Pantau perkembangan permintaan pasar."],
            "Kurang Laris": ["Tingkatkan promosi produk.", "Evaluasi strategi pemasaran.", "Pantau penjualan secara berkala."],
            "Sangat Rendah": ["Evaluasi produk dengan tingkat penjualan terendah.", "Pertimbangkan pemberian diskon/promosi.", "Kurangi pengadaan stok."]
        }
        
        # Urutan Tampil Rekomendasi: Laris -> Sedang -> Kurang Laris
        custom_rec_order = ["Sangat Laris", "Laris", "Sedang", "Kurang Laris", "Sangat Rendah"]
        summary_rec = summary.copy()
        summary_rec['Kategori'] = pd.Categorical(summary_rec['Kategori'], categories=custom_rec_order, ordered=True)
        summary_rec = summary_rec.sort_values('Kategori').reset_index(drop=True)

        inv_label_map = {v: k for k, v in corrected_label_map.items()}
        for _, row in summary_rec.iterrows():
            kategori = row['Kategori']
            c = get_color(kategori)
            cluster_id = inv_label_map.get(kategori, "?")
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
    # 5. JARAK EUCLIDEAN KE CENTROID (KALKULASI DAN URUTAN DIJAMIN TEPAT)
    # =========================================================================
    st.markdown("---")
    st.markdown("<div class='section-title'>📐 Jarak Euclidean ke Centroid</div>", unsafe_allow_html=True)
    
    # Ambil data ter-normalisasi
    # Mengambil fitur numerik yang dipakai untuk clustering (misal Qty_2022_2025_norm / Qty)
    norm_col = [c for c in df_clustered.columns if 'norm' in c or 'Qty' in c or c != 'Nama Barang']
    if norm_col:
        X_vals = df_clustered[norm_col[0]].values.reshape(-1, 1)
    else:
        X_vals = scaler.transform(df_agg[['Qty_2022_2025']])

    # Hitung jarak asli ke tiap centroid
    all_distances = np.abs(X_vals - model.cluster_centers_.T)

    df_dist_display = pd.DataFrame({'Nama Barang': df_clustered['Nama Barang']})

    # Urutan Kategori yang Diinginkan: Kurang Laris -> Sedang -> Laris
    target_categories = ["Kurang Laris", "Sedang", "Laris"]
    inv_map = {v: k for k, v in corrected_label_map.items()}

    for cat in target_categories:
        if cat in inv_map:
            cid = inv_map[cat]
            df_dist_display[f"Jarak ke Centroid ({cat})"] = np.round(all_distances[:, cid], 4)

    st.dataframe(df_dist_display.head(50), use_container_width=True)

    st.markdown("---")
    csv = df_full[['Nama Barang', 'Qty_2022_2025', 'Kategori']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Clustering (.csv)", data=csv, 
                       file_name=f"hasil_asri_mart_{datetime.now().strftime('%Y%m%d')}.csv", 
                       use_container_width=True, type="primary")
