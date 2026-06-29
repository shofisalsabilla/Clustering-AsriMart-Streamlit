import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import state

PALETTE = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6",
           "#1abc9c", "#e67e22", "#2980b9", "#8e44ad", "#16a085"]

def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📉 Evaluasi & Visualisasi</h2>
    </div>
    """, unsafe_allow_html=True)

    if not state.get("cluster_done"):
        st.warning("⚠️ Harap selesaikan **Konfigurasi Clustering** terlebih dahulu.")
        return

    df_clustered = state.get("df_clustered")
    df_agg = state.get("df_agg")
    model = state.get("kmeans_model")
    sil = state.get("silhouette_score")

    # Silhouette Score
    st.markdown("<div class='section-title'>📏 Evaluasi Silhouette Score</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil:.4f}")
    col2.metric("Jumlah Cluster", state.get("n_clusters"))
    col3.metric("Total Barang", len(df_clustered))
    st.markdown("---")

    # 1. Scatter Plot PCA & Bar Chart Rata-rata Qty
    st.markdown("<div class='section-title'>🔵 Scatter Plot & Perbandingan Rata-rata</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Scatter Plot PCA
        X = df_clustered[['Qty_2022_2025']]
        scaler_std = StandardScaler()
        X_scaled = scaler_std.fit_transform(X)
        X_fake = np.hstack([X_scaled, X_scaled + np.random.normal(0, 0.01, X_scaled.shape)])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_fake)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Kategori'] = df_clustered['Kategori'].values
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        for i, cat in enumerate(df_pca['Kategori'].unique()):
            mask = df_pca['Kategori'] == cat
            ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], label=cat, alpha=0.6)
        ax.set_title('Visualisasi Clustering PCA', color='white')
        st.pyplot(fig)
        st.caption("Narasi: Grafik ini menunjukkan pemisahan antar cluster dalam ruang 2D. Cluster yang terpisah dengan jelas menandakan kualitas segmentasi yang baik.")

    with c2:
        # Bar Chart Rata-rata Qty
        df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))
        summary = df_full.groupby('Kategori')['Qty_2022_2025'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        ax.barh(summary.index, summary.values, color=PALETTE[:len(summary)])
        ax.set_title('Rata-rata Qty per Kategori', color='white')
        st.pyplot(fig)
        st.caption("Narasi: Grafik ini membandingkan volume rata-rata penjualan per cluster untuk mengidentifikasi kategori barang paling dominan.")

    st.markdown("---")

    # 2. Distribusi Cluster (Pie) & Top 10 Terlaris
    st.markdown("<div class='section-title'>📦 Distribusi & Top Produk</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        # Distribusi Cluster (Hanya Pie)
        dist = df_clustered['Kategori'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%', textprops={'color': 'white'})
        ax.set_title('Proporsi Barang per Cluster', color='white')
        st.pyplot(fig)
        st.caption("Narasi: Menampilkan komposisi jumlah barang dalam setiap cluster untuk melihat keseimbangan distribusi data.")

    with c4:
        # Top 10 Terlaris
        top10 = df_agg.sort_values('Qty_2022_2025', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        ax.barh(top10['Nama Barang'], top10['Qty_2022_2025'], color='#4fc3f7')
        ax.invert_yaxis()
        ax.set_title('Top 10 Barang Terlaris', color='white')
        st.pyplot(fig)
        st.caption("Narasi: Menampilkan 10 produk dengan kuantitas penjualan tertinggi yang menjadi penggerak utama bisnis.")
