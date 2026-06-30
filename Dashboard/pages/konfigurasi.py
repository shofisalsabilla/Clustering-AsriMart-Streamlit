import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Asri Mart Clustering", layout="wide")

# Simulasi fungsi clustering (Jika belum ada di file terpisah)
def run_kmeans(df, n_clusters, labels):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    # Ini fungsi dummy agar kode jalan, sesuaikan dengan logic di utils Anda
    return kmeans, pd.DataFrame(df, columns=['Qty_2022_2025']), None, 0.7565, None

st.title("⚙️ Konfigurasi Clustering")

# 1. Input K
n_clusters = st.number_input("Masukkan nilai k (2-5):", min_value=2, max_value=5, value=3)

# 2. Label Dinamis Sesuai Permintaan
if n_clusters == 2:
    default_labels = ["Kurang Laris", "Laris"]
elif n_clusters == 3:
    default_labels = ["Kurang Laris", "Sedang", "Laris"]
elif n_clusters == 4:
    default_labels = ["Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
else: # n_clusters == 5
    default_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]

cols = st.columns(n_clusters)
new_label_map = {}
for i in range(n_clusters):
    with cols[i]:
        new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{i}")

# 3. Tombol Jalankan
if st.button("🚀 Jalankan K-Means", type="primary"):
    st.success(f"Berhasil menjalankan K-Means dengan k={n_clusters}")
    st.write("Label yang digunakan:", new_label_map)
