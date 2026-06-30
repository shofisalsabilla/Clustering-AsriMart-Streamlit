with col_kanan:
        st.markdown("<div class='section-title'>🎯 Posisi Centroid</div>", unsafe_allow_html=True)
        inv_label_map = {v: k for k, v in label_map.items()}
        centroid_data = []
        
        # Iterasi sesuai urutan summary agar konsisten
        for _, row in summary.iterrows():
            kategori = row['Kategori']
            cid = inv_label_map.get(kategori)
            if cid is not None:
                # Mengambil nilai mentah (ter-normalisasi) langsung dari model
                centroid_val = model.cluster_centers_[cid][0] 
                
                centroid_data.append({
                    "Kategori": kategori, 
                    "Centroid": f"[{centroid_val:.8f}]" # Format 8 angka di belakang koma
                })
        st.table(pd.DataFrame(centroid_data))
