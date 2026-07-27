import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils import state, clustering

def show():
    state.init_state()

    st.markdown("## ⚙️ Konfigurasi Clustering")

    if not state.get("upload_done"):
        st.warning("⚠️ Harap selesaikan **Upload & Preprocessing** terlebih dahulu.")
        return

    df_scaled = state.get("df_scaled")

    # 1. Metode Elbow
    st.markdown("### 📉 Metode Elbow")
    st.info("""
    Metode Elbow menunjukkan titik di mana penurunan nilai SSE (Inertia) mulai melambat secara signifikan, 
    membentuk sudut seperti **"siku"**. Nilai k pada titik siku tersebut dianggap sebagai jumlah 
    cluster yang optimal untuk data tersebut.
    """)
    
    col_control, col_graph = st.columns([1, 2])
    with col_control:
        k_max = st.slider("Batas maksimum k:", min_value=5, max_value=15, value=10)
        if st.button("🔍 Hitung Elbow Method"):
            wcss_data = clustering.compute_elbow(df_scaled, k_max)
            state.set("wcss", wcss_data)
            
    with col_graph:
        wcss = state.get("wcss")
        if wcss is not None:
            k_values = range(1, len(wcss) + 1)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_values, wcss, marker='o', linestyle='-', color='#007acc', linewidth=2, markersize=6)
            ax.set_title("Elbow Method Analysis", fontsize=12)
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Inertia (SSE)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.warning("Tekan tombol 'Hitung Elbow Method' untuk menampilkan grafik.")

    # 2. Input Label Kategori
    st.markdown("---")
    n_clusters = st.number_input(
        "Masukkan nilai k ", 
        min_value=2, 
        max_value=5, 
        value=state.get("n_clusters") or 3
    )
    state.set("n_clusters", n_clusters)
    
    if n_clusters == 2:
        default_labels = ["Kurang Laris", "Laris"]
    elif n_clusters == 3:
        default_labels = ["Kurang Laris", "Sedang", "Laris"]
    elif n_clusters == 4:
        default_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris"]
    else: 
        default_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    
    cols = st.columns(n_clusters)
    new_label_map = {}
    for i in range(n_clusters):
        with cols[i]:
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{n_clusters}_{i}")

    # 3. Jalankan K-Means
    if st.button("🚀 Jalankan K-Means", type="primary", use_container_width=True):
        try:
            model, df_clustered, df_dist, sil, centroids = clustering.run_kmeans(df_scaled, n_clusters, new_label_map)
            
            # Pengurutan agar cluster dengan rata-rata Qty terendah mendapat urutan pertama
            cluster_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
            mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(cluster_means.items())}
            
            df_clustered['Cluster'] = df_clustered['Cluster'].map(mapping)
            df_clustered['Kategori'] = df_clustered['Cluster'].map(new_label_map)
            
            # Simpan variabel ke session state
            state.set("kmeans_model", model)
            state.set("df_clustered", df_clustered)
            state.set("df_distances", df_dist)
            state.set("silhouette_score", sil)
            state.set("cluster_labels", new_label_map)
            state.set("centroids", centroids)
            state.set("cluster_done", True)
            
            st.success(f"✅ Berhasil! Hasil clustering telah diproses. Silhouette Score: {sil:.4f}")
            
            # Tampilkan informasi Centroid Akhir
            st.markdown("### 🎯 Posisi Centroid Akhir")
            sorted_centroids = sorted(centroids, key=lambda x: x[0])
            centroid_df = pd.DataFrame({
                "Kategori": [new_label_map[i] for i in range(n_clusters)],
                "Centroid (Scaled)": [f"[{c[0]:.8f}]" for c in sorted_centroids]
            })
            st.table(centroid_df)

        except Exception as e:
            st.error(f"Error saat menjalankan K-Means: {e}")
