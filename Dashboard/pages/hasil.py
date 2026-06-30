import streamlit as st
import pandas as pd
from datetime import datetime
from utils import state

def show():
    state.init_state()

    if not state.get("cluster_done"):
        st.warning("⚠️ Jalankan K-Means terlebih dahulu.")
        return

    df_clustered = state.get("df_clustered")
    label_map = state.get("cluster_labels")

    # Data Processing
    df_means = df_clustered.groupby('Kategori')['Qty_2022_2025'].mean().sort_values()
    
    # 1. Ringkasan & Strategi
    col_kiri, col_kanan = st.columns([1, 1])
    
    with col_kanan:
        st.markdown("<div class='section-title'>📝 Ringkasan Eksekutif & Strategi</div>", unsafe_allow_html=True)
        
        # Logika strategi otomatis berdasarkan posisi index
        total_k = len(df_means)
        for i, (kategori, avg_qty) in enumerate(df_means.items()):
            if i == 0:
                strategi = ["Evaluasi ulang stok.", "Berikan diskon untuk menghabiskan stok.", "Pantau perputaran barang."]
            elif i == total_k - 1:
                strategi = ["Jaga ketersediaan stok.", "Produk prioritas pemasaran.", "Pertahankan kualitas."]
            else:
                strategi = ["Pertahankan stok stabil.", "Lakukan promo berkala."]
            
            st.markdown(f"""
            <div style='background:#f9f9f9; padding:10px; border-radius:8px; border-left:4px solid #3498db; margin-bottom:8px;'>
                <strong>{kategori}</strong>
                <ul style='margin-bottom:0;'>{''.join([f'<li>{s}</li>' for s in strategi])}</ul>
            </div>
            """, unsafe_allow_html=True)

    with col_kiri:
        st.markdown("<div class='section-title'>📋 Tabel Hasil Pengelompokan</div>", unsafe_allow_html=True)
        # Filter kategori dinamis
        all_cats = ["Semua"] + list(df_means.index)
        selected_cat = st.selectbox("Filter Kategori:", all_cats)
        
        df_show = df_clustered[['Nama Barang', 'Qty_2022_2025', 'Kategori']].copy()
        if selected_cat != "Semua":
            df_show = df_show[df_show['Kategori'] == selected_cat]
        
        st.dataframe(df_show.sort_values('Qty_2022_2025', ascending=False), use_container_width=True)

    # 6. Download
    st.markdown("---")
    csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="hasil_clustering.csv", mime="text/csv")
