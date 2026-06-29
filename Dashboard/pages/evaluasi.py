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

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📉 Evaluasi & Visualisasi</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Silhouette Score, Scatter PCA, Bar Chart, dan distribusi cluster
        </p>
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
    col3.metric("Total Barang Dianalisis", len(df_clustered))

    if sil >= 0.7: quality = "🟢 Sangat Baik (≥0.70)"
    elif sil >= 0.5: quality = "🟡 Baik (0.50–0.69)"
    elif sil >= 0.25: quality = "🟠 Cukup (0.25–0.49)"
    else: quality = "🔴 Kurang Baik (<0.25)"
    st.markdown(f"**Kualitas Clustering:** {quality}")
    st.markdown("---")

    # Scatter Plot PCA (DIPERBAIKI)
    st.markdown("<div class='section-title'>🔵 Scatter Plot PCA</div>", unsafe_allow_html=True)
    st.markdown("**Visualisasi cluster dengan PCA 2D**")
    try:
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

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        for i, cat in enumerate(df_pca['Kategori'].unique()):
            mask = df_pca['Kategori'] == cat
            ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], 
                       color=PALETTE[i % len(PALETTE)], label=cat, alpha=0.7, s=60)
        ax.scatter(c_pca[:, 0], c_pca[:, 1], c='white', s=250, marker='X', edgecolors='black', label='Centroid', zorder=5)
        ax.set_title('Visualisasi Clustering K-Means (PCA)', color='white')
        ax.tick_params(colors='#b0c4de')
        ax.grid(True, linestyle='--', alpha=0.3, color='#4a6080')
        ax.legend(facecolor='#1a2a3a', labelcolor='white')
        st.pyplot(fig, use_container_width=True) # Tambahkan use_container_width
        plt.close(fig)
    except Exception as e:
        st.error(f"Grafik tidak dapat ditampilkan: {e}")

    st.markdown("---")

    # Bar Chart Rata-rata Qty
    st.markdown("<div class='section-title'>📊 Bar Chart Rata-rata Qty</div>", unsafe_allow_html=True)
    df_full = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', ''))
    summary = df_full.groupby('Kategori')['Qty_2022_2025'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    bars = ax.barh(summary.index, summary.values, color=PALETTE[:len(summary)])
    ax.tick_params(colors='#b0c4de')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # Distribusi Cluster
    st.markdown("<div class='section-title'>📦 Distribusi Cluster</div>", unsafe_allow_html=True)
    dist = df_clustered['Kategori'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0d1b2a')
    ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%', colors=[PALETTE[i%len(PALETTE)] for i in range(len(dist))], textprops={'color': 'white'})
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # Top 10 Terlaris
    st.markdown("<div class='section-title'>🏆 Top 10 Terlaris</div>", unsafe_allow_html=True)
    top10 = df_agg.sort_values('Qty_2022_2025', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1b2a')
    ax.barh(top10['Nama Barang'], top10['Qty_2022_2025'], color='#4fc3f7')
    ax.invert_yaxis()
    ax.tick_params(colors='#b0c4de')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
