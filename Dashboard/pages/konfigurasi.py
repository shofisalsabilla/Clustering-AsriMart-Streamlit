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

    # Elbow Method Section
    st.markdown("<div class='section-title'>📉 Metode Elbow</div>", unsafe_allow_html=True)
    col_control, col_graph = st.columns([1, 2])

    with col_control:
        k_max = st.slider("Batas maksimum k:", min_value=5, max_value=15, value=10)
        if st.button("🔍 Hitung Elbow Method", use_container_width=True):
            wcss = clustering.compute_elbow(df_scaled, k_max)
            state.set("wcss", wcss)

    with col_graph:
        wcss = state.get("wcss")
        if wcss:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(range(1, len(wcss) + 1), wcss, marker='o', color='#4fc3f7')
            st.pyplot(fig)

    # Konfigurasi Jumlah Cluster (k)
    st.markdown("---")
    st.markdown("<div class='section-title'>🎛️ Pilih Jumlah Cluster (k)</div>", unsafe_allow_html=True)
    n_clusters = st.number_input("Masukkan nilai k:", min_value=2, max_value=10, value=state.get("n_clusters") or 3)
    state.set("n_clusters", n_clusters)

    # Label Cluster Dinamis
    st.markdown("<div class='section-title'>🏷️ Tentukan Nama Label Cluster</div>", unsafe_allow_html=True)
    
    # List label default yang bisa menampung hingga 10 cluster
    default_labels = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi", 
                      "Cluster 6", "Cluster 7", "Cluster 8", "Cluster 9", "Cluster 10"]

    cols = st.columns(min(n_clusters, 5))
    new_label_map = {}
    
    for i in range(n_clusters):
        with cols[i % min(n_clusters, 5)]:
            # Mengambil nilai dari state jika sudah ada agar tidak reset saat interaksi lain
            prev_labels = state.get("cluster_labels")
            default_val = prev_labels.get(i, default_labels[i]) if isinstance(prev_labels, dict) else default_labels[i]
            
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_val, key=f"label_{i}")

    state.set("cluster_labels", new_label_map)

    # Jalankan
    if st.button("🚀 Jalankan K-Means", type="primary", use_container_width=True):
        try:
            model, df_clustered, df_dist, sil, dyn_map = clustering.run_kmeans(df_scaled, n_clusters, new_label_map)
            state.set("kmeans_model", model)
            state.set("df_clustered", df_clustered)
            state.set("df_distances", df_dist)
            state.set("silhouette_score", sil)
            state.set("cluster_labels", dyn_map)
            state.set("cluster_done", True)
            st.success(f"✅ Berhasil! Silhouette: {sil:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
