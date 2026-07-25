with col_rek:
        st.markdown("<div class='section-title'>📝 Rekomendasi/Strategi</div>", unsafe_allow_html=True)
        REKOMENDASI = {
            "Sangat Laris": ["Tingkatkan ketersediaan stok untuk menghindari kehabisan.", "Jadikan produk sebagai produk unggulan.", "Pertahankan strategi pemasaran yang efektif."],
            "Laris": ["Prioritaskan ketersediaan stok.", "Jadikan produk fokus pemasaran.", "Pertahankan kualitas produk dan layanan."],
            "Sedang": ["Pertahankan performa penjualan yang stabil.", "Lakukan promosi secara berkala.", "Pantau perkembangan permintaan pasar."],
            "Kurang Laris": ["Tingkatkan promosi produk.", "Evaluasi strategi pemasaran.", "Pantau penjualan secara berkala."],
            "Sangat Rendah": ["Evaluasi produk dengan tingkat penjualan terendah.", "Pertimbangkan pemberian diskon/promosi.", "Kurangi pengadaan stok."]
        }
        
        # Urutan urut dari Laris -> Sedang -> Kurang Laris
        custom_rec_order = ["Sangat Laris", "Laris", "Sedang", "Kurang Laris", "Sangat Rendah"]
        summary_rec = summary.copy()
        summary_rec['Kategori'] = pd.Categorical(summary_rec['Kategori'], categories=custom_rec_order, ordered=True)
        summary_rec = summary_rec.sort_values('Kategori').reset_index(drop=True)

        # Mapping ID Cluster khusus untuk tampilan kartu rekomendasi
        rec_cluster_map = {
            "Laris": 1,
            "Sedang": 2,
            "Kurang Laris": 0
        }

        for _, row in summary_rec.iterrows():
            kategori = row['Kategori']
            c = get_color(kategori)
            cluster_id = rec_cluster_map.get(kategori, "?")
            poin = REKOMENDASI.get(kategori, ["Pantau perkembangan berkala."])
            list_html = "".join([f"<li style='margin-bottom:4px; color:#000000;'>{p}</li>" for p in poin])
            
            st.markdown(f"""
            <div style='background:#ffffff; border-left:4px solid {c}; border-radius:10px; padding:14px 18px; margin-bottom:12px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-weight:700; color:{c}; font-size:1rem;'>{kategori}</div>
                    <div style='background:{c}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:bold;'>
                        Cluster {cluster_id}
                    </div>
                </div>
                <div style='color:#000000; font-size:0.85rem; margin-top:8px;'>
                    <ul style='margin:6px 0 0 18px; padding:0;'>{list_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
