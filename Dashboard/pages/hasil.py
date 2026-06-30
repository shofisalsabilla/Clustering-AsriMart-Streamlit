import streamlit as st
import pandas as pd
from utils import state

def show():
    state.init_state()
    if not state.get("cluster_done"):
        st.warning("⚠️ Jalankan K-Means terlebih dahulu.")
        return

    df_clustered = state.get("df_clustered")
    
    # Ambil urutan kategori yang benar berdasarkan rata-rata Qty agar stabil
    df_means = df_clustered.groupby(['Cluster', 'Kategori'])['Qty_2022_2025'].mean().reset_index().sort_values('Qty_2022_2025')
    
    st.markdown("## 📊 Hasil Clustering")
    col_kiri, col_kanan = st.columns([1, 1])

    with col_kanan:
        st.markdown("### 📝 Ringkasan Eksekutif & Strategi")
        total_k = len(df_means)
        for i, row in df_means.iterrows():
            kategori = row['Kategori']
            # Logika strategi dinamis
            if i == 0:
                strategi = ["Stok sangat rendah, segera evaluasi.", "Diskon habiskan stok."]
            elif i == total_k - 1:
                strategi = ["Prioritas stok utama.", "Maksimalkan promosi."]
            else:
                strategi = ["Pertahankan stok stabil.", "Pantau tren berkala."]
            
            st.markdown(f"""
            <div style='background:#f9f9f9; padding:10px; border-radius:8px; border-left:4px solid #3498db; margin-bottom:8px;'>
                <strong>{kategori}</strong> (Avg Qty: {row['Qty_2022_2025']:.1f})
                <ul>{''.join([f'<li>{s}</li>' for s in strategi])}</ul>
            </div>
            """, unsafe_allow_html=True)

    with col_kiri:
        st.markdown("### 📋 Tabel Hasil")
        all_cats = ["Semua"] + list(df_means['Kategori'].unique())
        selected_cat = st.selectbox("Filter Kategori:", all_cats)
        
        df_show = df_clustered[['Nama Barang', 'Qty_2022_2025', 'Kategori']]
        if selected_cat != "Semua":
            df_show = df_show[df_show['Kategori'] == selected_cat]
        
        st.dataframe(df_show.sort_values('Qty_2022_2025', ascending=False), use_container_width=True)

    # Download
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="hasil_clustering.csv", mime="text/csv")
