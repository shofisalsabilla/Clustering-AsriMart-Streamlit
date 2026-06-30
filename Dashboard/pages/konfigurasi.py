import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils import state, clustering

def show():
    state.init_state()

    st.markdown("<div class='main-header'><h2 style='margin:0; color:white;'>⚙️ Konfigurasi Clustering</h2></div>", unsafe_allow_html=True)

    if not state.get("upload_done"):
        st.warning("⚠️ Harap selesaikan **Upload & Preprocessing** terlebih dahulu.")
        return

    df_scaled = state.get("df_scaled")

    # 1. Elbow Method
    st.markdown("<div class='section-title'>📉 Metode Elbow</div>", unsafe_allow_html=True)
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

    # 2. Input Label
    st.markdown("---")
    n_clusters = st.number_input("Masukkan nilai k:", min_value=2, max_value=10, value=state.get("n_clusters") or 3)
    state.set("n_clusters", n_clusters)
    
    default_labels = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi", 
                      "Cluster 6", "Cluster 7", "Cluster 8", "Cluster 9", "Cluster 10"]
    
    cols = st.columns(min(n_clusters, 5))
    new_label_map = {}
    for i in range(n_clusters):
        with cols[i % min(n_clusters, 5)]:
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{i}")

    # 3. Jalankan K-Means dengan Pengurutan Otomatis
    if st.button("🚀 Jalankan K-Means", type="primary", use_container_width=True):
        try:
            model, df_clustered, df_dist, sil, _ = clustering.run_kmeans(df_scaled, n_clusters, new_label_map)
            
            # --- LOGIKA PENGURUTAN ---
            # Urutkan berdasarkan rata-rata Qty
            cluster_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
            mapping = {old_id: i for i, (old_id, _) in enumerate(cluster_means.items())}
            
            df_clustered['Cluster'] = df_clustered['Cluster'].map(mapping)
            df_clustered['Kategori'] = df_clustered['Cluster'].map(new_label_map)
            # -------------------------

            state.set("kmeans_model", model)
            state.set("df_clustered", df_clustered)
            state.set("df_distances", df_dist)
            state.set("silhouette_score", sil)
            state.set("cluster_labels", new_label_map)
            state.set("cluster_done", True)
            st.success(f"✅ Berhasil! Silhouette: {sil:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
