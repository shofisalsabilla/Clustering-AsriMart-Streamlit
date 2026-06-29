import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import state

PALETTE = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6",
           "#1abc9c", "#e67e22", "#2980b9", "#8e44ad", "#16a085"]

def show():
    state.init_state()

    st.markdown("<h2 style='text-align: center; color: white;'>📉 Evaluasi & Visualisasi Model</h2>", unsafe_allow_html=True)

    if not state.get("cluster_done"):
        st.warning("⚠️ Harap selesaikan **Konfigurasi Clustering** terlebih dahulu.")
        return

    df_clustered = state.get("df_clustered")
    df_agg = state.get("df_agg")
    model = state.get("kmeans_model")
    sil = state.get("silhouette_score")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil:.4f}")
    col2.metric("Jumlah Cluster", state.get("n_clusters"))
    col3.metric("Total Produk", len(df_clustered))
    st.markdown("---")

    # Fungsi pembantu untuk visualisasi agar konsisten
    def plot_container(title, description, plot_func):
        with st.container(border=True):
            st.subheader(title)
            plot_func()
            st.info(description)

    # ── ROW 1 ─────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>🔵 Scatter Plot PCA</div>", unsafe_allow_html=True)
        X = df_clustered[['Qty_2022_2025']]
        scaler_std = StandardScaler()
        X_scaled = scaler_std.fit_transform(X)
        X_fake = np.hstack([X_scaled, X_scaled + np.random.normal(0, 0.01, X_scaled.shape)])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_fake)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Kategori'] = df_clustered['Kategori'].values
        centroids = model.cluster_centers_
        c_scaled = scaler_std.transform(centroids)
        c_fake = np.hstack([c_scaled, c_scaled])
        c_pca = pca.transform(c_fake)
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
        for i, cat in enumerate(df_pca['Kategori'].unique()):
            mask = df_pca['Kategori'] == cat
            ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], color=PALETTE[i % len(PALETTE)], label=cat, alpha=0.7, s=60)
        ax.scatter(c_pca[:, 0], c_pca[:, 1], c='white', s=250, marker='X', edgecolors='black', label='Centroid', zorder=5)
        ax.tick_params(colors='#b0c4de'); ax.grid(True, linestyle='--', alpha=0.3, color='#4a6080')
        st.pyplot(fig); plt.close(fig)

        plot_container("🔵 Scatter Plot PCA", 
                       "Visualisasi ini menunjukkan pemisahan antar cluster dalam ruang 2D. Semakin jauh jarak antar kelompok, semakin baik kualitas segmentasi data Anda.", 
                       pca_plot)

    with c2:
        def bar_plot():
            df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))
            summary = df_full.groupby('Kategori')['Qty_2022_2025'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
            ax.barh(summary.index, summary.values, color=PALETTE[:len(summary)])
            ax.tick_params(colors='#b0c4de'); ax.grid(axis='x', linestyle='--', alpha=0.2)
            st.pyplot(fig); plt.close(fig)

        plot_container("📊 Rata-rata Qty per Cluster", 
                       "Grafik ini membandingkan intensitas penjualan rata-rata pada setiap cluster, membantu mengidentifikasi mana cluster 'Super Laris' dan mana yang 'Slow Moving'.", 
                       bar_plot)

    # ── ROW 2 ─────────
    c3, c4 = st.columns(2)

    with c3:
        def dist_plot():
            dist = df_clustered['Kategori'].value_counts()
            labels = [f"{idx}\n({val} produk)" for idx, val in dist.items()]
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
            ax.pie(dist.values, labels=labels, autopct='%1.1f%%', colors=PALETTE[:len(dist)], textprops={'color': 'white'})
            st.pyplot(fig); plt.close(fig)

        plot_container("📦 Proporsi Produk per Cluster", 
                       "Menampilkan distribusi jumlah produk di setiap kategori. Grafik ini memberikan gambaran seberapa seimbang pembagian segmentasi yang dihasilkan model.", 
                       dist_plot)

    with c4:
        def top_plot():
            top10 = df_agg.sort_values('Qty_2022_2025', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
            ax.barh(top10['Nama Barang'], top10['Qty_2022_2025'], color='#4fc3f7')
            ax.invert_yaxis(); ax.tick_params(colors='#b0c4de'); ax.grid(axis='x', linestyle='--', alpha=0.2)
            st.pyplot(fig); plt.close(fig)

        plot_container("🏆 Top 10 Produk Terlaris", 
                       "Daftar 10 produk dengan volume penjualan tertinggi selama periode 2022-2025. Produk dalam daftar ini adalah pilar utama pendapatan toko Anda.", 
                       top_plot)
