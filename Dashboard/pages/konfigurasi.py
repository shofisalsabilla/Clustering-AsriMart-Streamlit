# 2. Input Label dengan Logika Dinamis (Diperbarui sesuai urutan Anda)
    st.markdown("---")
    n_clusters = st.number_input(
        "Masukkan nilai k (2-5):", 
        min_value=2, 
        max_value=5, 
        value=state.get("n_clusters") or 3
    )
    state.set("n_clusters", n_clusters)
    
    # Logika label yang diperbarui agar k 2-4 sesuai permintaan
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
