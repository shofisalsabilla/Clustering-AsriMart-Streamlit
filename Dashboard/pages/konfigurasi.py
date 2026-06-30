import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils import state, clustering

def show():
    state.init_state()

    # Perbaikan: Gunakan st.container dengan class untuk styling jika memungkinkan,
    # namun agar aman dari TypeError, kita gunakan penulisan Markdown yang bersih.
    st.markdown("## ⚙️ Konfigurasi Clustering")

    if not state.get("upload_done"):
        st.warning("⚠️ Harap selesaikan **Upload & Preprocessing** terlebih dahulu.")
        return

    df_scaled = state.get("df_scaled")

    # 1. Metode Elbow
    st.markdown("### 📉 Metode Elbow")
    col_control, col_graph = st.columns([1, 2])
    with col_control:
        k_max = st.slider("Batas maksimum k:", min_value=5, max_value=15, value=10)
        if st.button("🔍 Hitung Elbow Method"):
            state.set("wcss", clustering.compute_elbow(df_scaled, k_max))
    with col_graph:
        if state.get("wcss"):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(range(1, len(state.get("wcss")) + 1), state.get("wcss"), marker='o', color='#4fc3f7')
            st.pyplot(fig)

    # 2. Input Label dengan Logika Dinamis (Koreksi Penamaan)
    st.markdown("---")
    n_clusters = st.number_input(
        "Masukkan nilai k (2-5):", 
        min_value=2, 
        max_value=5, 
        value=state.get("n_clusters") or 3
    )
    state.set("n_clusters", n_clusters)
    
    # Koreksi penamaan label sesuai permintaan
    if n_clusters == 2:
        default_labels = ["Kurang Laris", "Laris"]
    elif n_clusters == 3:
        default_labels = ["Kurang Laris", "Laris", "Sangat Laris"]
    elif n_clusters == 4:
        default_labels = ["Sangat Rendah", "Kurang Laris", "Laris", "Sangat Laris"]
    else: # n_clusters == 5
        default_labels = ["Sangat Rendah", "Rendah", "Kurang Laris", "Laris", "Sangat Laris"]
    
    cols = st.columns(min(n_clusters, 5))
    new_label_map = {}
    for i in range(n_clusters):
        with cols[i % min(n_clusters, 5)]:
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{i}")

    # 3. Jalankan K-Means dengan Pengurutan (Logic tetap konsisten)
    if st.button("🚀 Jalankan K-Means", type="primary", use_container_width=True):
        try:
            model, df_clustered, df_dist, sil, _ = clustering.run_kmeans(df_scaled, n_clusters, new_label_map)
            
            cluster_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
            mapping = {old_id: i for i, (old_id, _) in enumerate(cluster_means.items())}
            
            sorted_label_list = [new_label_map[old_id] for old_id, _ in cluster_means.items()]
            final_label_map = {i: label for i, label in enumerate(sorted_label_list)}
            
            df_clustered['Cluster'] = df_clustered['Cluster'].map(mapping)
            df_clustered['Kategori'] = df_clustered['Cluster'].map(final_label_map)
            
            state.set("kmeans_model", model)
            state.set("df_clustered", df_clustered)
            state.set("df_distances", df_dist)
            state.set("silhouette_score", sil)
            state.set("cluster_labels", final_label_map)
            state.set("cluster_done", True)
            
            st.success(f"✅ Berhasil! Cluster diurutkan berdasarkan rata-rata Qty. Silhouette: {sil:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
