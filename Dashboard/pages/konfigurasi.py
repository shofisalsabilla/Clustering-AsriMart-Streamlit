import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils import state, clustering

def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>⚙️ Konfigurasi Clustering</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Tentukan jumlah cluster optimal dengan Elbow Method
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not state.get("upload_done"):
        st.warning("⚠️ Harap selesaikan **Upload & Preprocessing** terlebih dahulu.")
        return

    df_scaled = state.get("df_scaled")

    # Elbow Method
    st.markdown("<div class='section-title'>📉 Metode Elbow (Menentukan k Optimal)</div>", unsafe_allow_html=True)

    k_max = st.slider("Batas maksimum k untuk analisis Elbow:", min_value=5, max_value=15, value=10)

    if st.button("🔍 Hitung Elbow Method", use_container_width=True):
        with st.spinner("Menghitung WCSS untuk setiap nilai k..."):
            wcss = clustering.compute_elbow(df_scaled, k_max)
            state.set("wcss", wcss)

    wcss = state.get("wcss")
    if wcss:
        # Grafik diperkecil dengan figsize=(7, 3)
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')

        K = range(1, len(wcss) + 1)
        ax.plot(K, wcss, marker='o', linestyle='--', color='#4fc3f7', linewidth=2, markersize=8)
        ax.fill_between(K, wcss, alpha=0.1, color='#4fc3f7')

        ax.set_xlabel('Jumlah Cluster (k)', color='#b0c4de', fontsize=9)
        ax.set_ylabel('SSE (Inertia)', color='#b0c4de', fontsize=9)
        ax.set_title('Grafik Metode Elbow', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#b0c4de', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#2a4a7f')
        ax.grid(True, linestyle='--', alpha=0.3, color='#4a6080')
        ax.set_xticks(list(K))

        st.pyplot(fig)
        plt.close(fig)
        st.caption("💡 Pilih nilai k di titik 'siku' — dimana penurunan WCSS mulai melambat.")

    # Konfigurasi k
    st.markdown("---")
    st.markdown("<div class='section-title'>🎛️ Pilih Jumlah Cluster (k)</div>", unsafe_allow_html=True)

    n_clusters = st.number_input(
        "Masukkan nilai k (jumlah cluster):",
        min_value=2, max_value=10,
        value=state.get("n_clusters") or 3,
        step=1
    )
    state.set("n_clusters", n_clusters)

    # Label cluster
    st.markdown("<div class='section-title'>🏷️ Tentukan Nama Label Cluster</div>", unsafe_allow_html=True)
    st.caption("Label akan ditetapkan dari cluster rata-rata terendah ke tertinggi secara otomatis.")

    default_labels = ["Kurang Laris", "Sedang", "Laris", "Sangat Laris",
                      "Cluster 5", "Cluster 6", "Cluster 7", "Cluster 8",
                      "Cluster 9", "Cluster 10"]

    label_map = {}
    cols = st.columns(min(n_clusters, 5))
    for i in range(n_clusters):
        with cols[i % min(n_clusters, 5)]:
            label = st.text_input(
                f"Rank {i+1} (terendah → tertinggi):",
                value=state.get("cluster_labels").get(i, default_labels[i]) if isinstance(state.get("cluster_labels"), dict) else default_labels[i],
                key=f"label_{i}"
            )
            label_map[i] = label

    # Simpan & Jalankan
    st.markdown("---")
    if st.button("🚀 Jalankan K-Means Clustering", type="primary", use_container_width=True):
        with st.spinner(f"Menjalankan K-Means dengan k={n_clusters}..."):
            try:
                model, df_clustered, df_dist, sil, dyn_map = clustering.run_kmeans(
                    df_scaled, n_clusters, label_map
                )
                state.set("kmeans_model", model)
                state.set("df_clustered", df_clustered)
                state.set("df_distances", df_dist)
                state.set("silhouette_score", sil)
                state.set("cluster_labels", dyn_map)
                state.set("cluster_done", True)
                st.success(f"✅ K-Means selesai! Silhouette Score: **{sil:.4f}**", icon="✅")
                st.info("Lihat hasil di menu **📊 Hasil Clustering** dan **📉 Evaluasi & Visualisasi**.")
            except Exception as e:
                st.error(f"❌ Error saat clustering: {e}")
