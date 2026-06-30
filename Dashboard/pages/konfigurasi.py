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

    # 2. Input Label dengan Logika Dinamis (Batas k 2-5)
    st.markdown("---")
    n_clusters = st.number_input(
        "Masukkan nilai k (2-5):", 
        min_value=2, 
        max_value=5, 
        value=state.get("n_clusters") or 3
    )
    state.set("n_clusters", n_clusters)
    
    # Logika label dinamis disesuaikan dengan permintaan
    if n_clusters == 2:
        default_labels = ["Kurang Laris", "Laris"]
    elif n_clusters == 3:
        default_labels = ["Kurang Laris", "Sedang", "Laris"]
    elif n_clusters == 4:
        default_labels = ["Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    else: # n_clusters == 5
        default_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    
    cols = st.columns(min(n_clusters, 5))
    new_label_map = {}
    for i in range(n_clusters):
        with cols[i % min(n_clusters, 5)]:
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{i}")

    # 3. Jalankan K-Means dengan Pengurutan Otomatis
    if st.button("🚀 Jalankan K-Means", type="primary", use_container_width=True):
        try:
            # Jalankan K-Means
            model, df_clustered, df_dist, sil, _ = clustering.run_kmeans(df_scaled, n_clusters, new_label_map)
            
            # --- LOGIKA PENGURUTAN ---
            # Hitung rata-rata tiap cluster untuk menentukan urutan[cite: 1]
            cluster_means = df_clustered.groupby('Cluster')['Qty_2022_2025'].mean().sort_values()
            
            # Buat mapping dari ID cluster lama ke urutan baru (0 = terkecil, dst)[cite: 1]
            mapping = {old_id: i for i, (old_id, _) in enumerate(cluster_means.items())}
            
            # Urutkan label agar sesuai dengan urutan mean yang baru[cite: 1]
            sorted_labels = [new_label_map[old_id] for old_id, _ in cluster_means.items()]
            final_label_map = {i: label for i, label in enumerate(sorted_labels)}
            
            # Terapkan ke DataFrame[cite: 1]
            df_clustered['Cluster'] = df_clustered['Cluster'].map(mapping)
            df_clustered['Kategori'] = df_clustered['Cluster'].map(final_label_map)
            # -------------------------

            state.set("kmeans_model", model)
            state.set("df_clustered", df_clustered)
            state.set("df_distances", df_dist)
            state.set("silhouette_score", sil)
            state.set("cluster_labels", final_label_map) # Simpan map yang sudah urut[cite: 1]
            state.set("cluster_done", True)
            
            st.success(f"✅ Berhasil! Silhouette: {sil:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
