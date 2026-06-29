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

    # Membuat dua kolom: [1, 2] berarti kiri lebih sempit, kanan lebih lebar
    col_control, col_graph = st.columns([1, 2])

    with col_control:
        k_max = st.slider("Batas maksimum k untuk analisis Elbow:", min_value=5, max_value=15, value=10)

        if st.button("🔍 Hitung Elbow Method", use_container_width=True):
            with st.spinner("Menghitung WCSS untuk setiap nilai k..."):
                wcss = clustering.compute_elbow(df_scaled, k_max)
                state.set("wcss", wcss)
        
        st.caption("💡 Pilih nilai k di titik 'siku' — dimana penurunan WCSS mulai melambat.")

    with col_graph:
        wcss = state.get("wcss")
        if wcss:
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor('#0d1b2a')
            ax.set_facecolor('#0d1b2a')

            K = range(1, len(wcss) + 1)
            ax.plot(K, wcss, marker='o', linestyle='--', color='#4fc3f7', linewidth=1.5, markersize=5)
            ax.fill_between(K, wcss, alpha=0.1, color='#4fc3f7')

            ax.set_xlabel('Jumlah Cluster (k)', color='#b0c4de', fontsize=8)
            ax.set_ylabel('SSE', color='#b0c4de', fontsize=8)
            ax.set_title('Grafik Metode Elbow', color='white', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#b0c4de', labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_color('#2a4a7f')
            ax.grid(True, linestyle='--', alpha=0.2, color='#4a6080')
            ax.set_xticks(list(K))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Menambahkan narasi penjelasan
            st.markdown("""
<div style='background-color:#ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #4fc3f7; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
    <p style='font-size: 0.9rem; color: #000000; margin: 0;'>
                    <strong>Cara membaca grafik:</strong> Metode Elbow menunjukkan titik di mana penurunan 
                    nilai SSE (Inertia) mulai melambat secara signifikan, membentuk sudut seperti <strong>"siku"</strong>. 
                    Nilai k pada titik siku tersebut dianggap sebagai jumlah cluster yang optimal untuk data tersebut.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
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
