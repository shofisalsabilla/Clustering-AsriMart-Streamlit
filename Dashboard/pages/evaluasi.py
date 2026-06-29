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

    # ... [Header dan Metric tetap sama] ...
    
    # ── ROW 1 & 2: Pengaturan layout seragam ─────────
    # Kita gunakan set ukuran (8, 5) untuk semua grafik agar konsisten
    FIGSIZE = (8, 5)

    col1, col2 = st.columns(2)

    # 1. Scatter Plot PCA
    with col1:
        st.markdown("<div class='section-title'>🔵 Scatter Plot PCA</div>", unsafe_allow_html=True)
        # ... [Proses PCA tetap sama] ...
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
        # ... [Plotting logic] ...
        st.pyplot(fig); plt.close(fig)

    # 2. Bar Chart Rata-rata Qty
    with col2:
        st.markdown("<div class='section-title'>📊 Bar Chart Rata-rata Qty</div>", unsafe_allow_html=True)
        # ... [Proses Bar Chart tetap sama] ...
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
        # ... [Plotting logic] ...
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    col3, col4 = st.columns(2)

    # 3. Distribusi Cluster (Pie Chart)
    with col3:
        st.markdown("<div class='section-title'>📦 Distribusi Cluster</div>", unsafe_allow_html=True)
        dist = df_clustered['Kategori'].value_counts()
        labels = [f"{idx}\n({val} prod)" for idx, val in dist.items()]
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
        ax.pie(dist.values, labels=labels, autopct='%1.1f%%', colors=PALETTE[:len(dist)], textprops={'color': 'white', 'fontsize': 9})
        st.pyplot(fig); plt.close(fig)

    # 4. Top 10 Terlaris
    with col4:
        st.markdown("<div class='section-title'>🏆 Top 10 Terlaris</div>", unsafe_allow_html=True)
        top10 = df_agg.sort_values('Qty_2022_2025', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
        bars = ax.barh(top10['Nama Barang'], top10['Qty_2022_2025'], color='#4fc3f7', edgecolor='none')
        # ... [Text styling agar tetap rapi di ukuran yang sama] ...
        ax.invert_yaxis()
        ax.tick_params(colors='#b0c4de'); ax.grid(axis='x', linestyle='--', alpha=0.3, color='#4a6080')
        st.pyplot(fig); plt.close(fig)
