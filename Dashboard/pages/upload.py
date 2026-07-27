import streamlit as st
import pandas as pd
import utils.clustering as clustering

def show():
    st.markdown('<div class="main-header"><h2>📂 Upload & Preprocessing Data</h2><p>Unggah file transaksi toko (.xlsx atau .csv) untuk diproses ke dalam tahap clustering.</p></div>', unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader("Pilih file dataset (Excel / CSV)", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            # Membaca file
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            st.success("✅ File berhasil diunggah!")
            
            # Validasi Kolom
            required_columns = {'Nama Barang', 'Qty'}
            if not required_columns.issubset(set(df_raw.columns)):
                st.error(f"❌ Format file salah! File wajib memiliki kolom: **{', '.join(required_columns)}**")
            else:
                # 1. Jalankan Preprocessing
                df_cleaned, df_agg, df_scaled, scaler = clustering.preprocess(df_raw)
                
                # 2. Simpan ke Session State agar bisa dipakai di halaman lain (Hasil, Evaluasi, dll)
                st.session_state['df_raw'] = df_raw
                st.session_state['df_cleaned'] = df_cleaned
                st.session_state['df_agg'] = df_agg
                st.session_state['df_scaled'] = df_scaled
                st.session_state['scaler'] = scaler
                
                # 3. Tampilkan Ringkasan Metrics
                st.markdown('<div class="section-title">📊 Ringkasan Data</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Total Baris Mentah", f"{len(df_raw):,}")
                col2.metric("Jumlah Barang Unik", f"{len(df_agg):,}")
                col3.metric("Total Qty Terjual", f"{int(df_agg['Total_Qty'].sum()):,}")
                
                # 4. Preview Data
                st.markdown('<div class="section-title">🔍 Preview Data</div>', unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["📄 Data Mentah", "🧹 Data Hasil Agregasi", "📏 Data Normalisasi (MinMax)"])
                
                with tab1:
                    st.write("Data asli dari file yang diunggah:")
                    st.dataframe(df_raw.head(100), use_container_width=True)
                    
                with tab2:
                    st.write("Data setelah pembersihan dan pembentukan `Total_Qty` per barang:")
                    st.dataframe(df_agg, use_container_width=True)
                    
                with tab3:
                    st.write("Data setelah dilakukan normalisasi ke rentang 0 - 1:")
                    st.dataframe(df_scaled, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            
    elif 'df_agg' in st.session_state:
        # Menampilkan status jika data sudah ada di session state sebelumnya
        st.info("ℹ️ Data transaksi sudah diunggah sebelumnya dan tersimpan di memori.")
        df_agg = st.session_state['df_agg']
        
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Barang Unik", f"{len(df_agg):,}")
        col2.metric("Total Qty Terjual", f"{int(df_agg['Total_Qty'].sum()):,}")
        
        st.dataframe(df_agg, use_container_width=True)
    else:
        st.info("Silakan unggah file transaksi `.xlsx` atau `.csv` untuk memulai.")
