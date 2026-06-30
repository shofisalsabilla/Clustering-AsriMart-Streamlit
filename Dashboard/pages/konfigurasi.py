# 2. Input Label dengan Logika Dinamis (Diperbarui)
    st.markdown("---")
    n_clusters = st.number_input(
        "Masukkan nilai k (2-5):", 
        min_value=2, 
        max_value=5, 
        value=state.get("n_clusters") or 3
    )
    state.set("n_clusters", n_clusters)
    
    # Logika label yang dikunci untuk k=2 hingga k=5
    if n_clusters == 2:
        default_labels = ["Kurang Laris", "Laris"]
    elif n_clusters == 3:
        default_labels = ["Kurang Laris", "Sedang", "Laris"]
    elif n_clusters == 4:
        default_labels = ["Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    else: # n_clusters == 5
        default_labels = ["Sangat Rendah", "Kurang Laris", "Sedang", "Laris", "Sangat Laris"]
    
    # Menampilkan input teks sesuai dengan jumlah cluster yang dipilih
    cols = st.columns(n_clusters) # Menggunakan n_clusters agar kolom sesuai jumlah
    new_label_map = {}
    for i in range(n_clusters):
        with cols[i]:
            new_label_map[i] = st.text_input(f"Rank {i+1}:", value=default_labels[i], key=f"label_{i}")
