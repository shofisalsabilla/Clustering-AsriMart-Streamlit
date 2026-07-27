import streamlit as st
import pandas as pd
import utils.clustering as clustering

st.set_page_config(page_title="Upload Data - Asri Mart", layout="wide")

st.title("📥 Upload & Preprocessing Data")
st.write("Unggah file transaksi toko (`.xlsx` atau `.csv`) untuk diproses ke dalam tahap clustering.")

# File Uploader
uploaded_file = st.file_uploader("Pilih file dataset", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # Membaca file sesuai ekstensi
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        st.success("File berhasil diunggah!")
        
        # Validasi Kolom Wajib
        required_columns = {'Nama Barang', 'Qty'}
        if not required_columns.issubset(set(df_raw.columns)):
            st.error(f"Format file salah! File harus memiliki kolom: **{', '.join(required_columns)}**")
        else:
            # 1. Jalankan Preprocessing
            df_cleaned, df_agg, df_scaled, scaler = clustering.preprocess(df_raw)
            
            # 2. Simpan ke Session State agar bisa diakses di halaman/tab lain
            st.session_state['df_raw'] = df_raw
            st.session_state['df_cleaned'] = df_cleaned
            st.session_state['df_agg'] = df_agg
            st.session_state['df_scaled'] = df_scaled
            st.session_state['scaler'] = scaler
            
            # 3. Tampilkan Ringkasan Metrics
            st.markdown("---")
            st.subheader("📊 Ringkasan Data")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Total Baris Mentah", f"{len(df_raw):,}")
            col2.metric("Jumlah Barang Unik", f"{len(df_agg):,}")
            col3.metric("Total Qty Terjual", f"{int(df_agg['Total_Qty'].sum()):,}")
            
            # 4. Preview Data dalam Tab
            tab1, tab2, tab3 = st.tabs(["📄 Data Mentah", "🧹 Data Hasil Agregasi", "📏 Data Scaled (MinMax)"])
            
            with tab1:
                st.write("Preview data asli dari file yang diunggah:")
                st.dataframe(df_raw.head(100), use_container_width=True)
                
            with tab2:
                st.write("Data setelah dibersihkan dan dijumlahkan per barang:")
                st.dataframe(df_agg, use_container_width=True)
                
            with tab3:
                st.write("Data setelah dinormalisasi dengan MinMax Scaler:")
                st.dataframe(df_scaled, use_container_width=True)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah dataset transaksi terlebih dahulu.")
