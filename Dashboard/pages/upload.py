import streamlit as st
import pandas as pd
import utils.clustering as clustering
from utils import state

def show():
    # Inisialisasi state awal
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📂 Upload & Preprocessing Data</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Unggah file transaksi toko (.xlsx atau .csv) untuk diproses ke dalam tahap clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
            
            # Validasi Kolom Wajib
            required_columns = {'Nama Barang', 'Qty'}
            if not required_columns.issubset(set(df_raw.columns)):
                st.error(f"❌ Format file salah! File wajib memiliki kolom: **{', '.join(required_columns)}**")
            else:
                # 1. Preprocessing Data
                df_cleaned, df_agg, df_scaled, scaler = clustering.preprocess(df_raw)
                
                # 2. Simpan Data ke Session State
                state.set("df_raw", df_raw)
                state.set("df_cleaned", df_cleaned)
                state.set("df_agg", df_agg)
                state.set("df_scaled", df_scaled)
                state.set("scaler", scaler)
                
                # 3. Tampilkan Ringkasan Metrics
                st.markdown("<div class='section-title'>📊 Ringkasan Data</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Baris Mentah", f"{len(df_raw):,}")
                col2.metric("Jumlah Barang Unik", f"{len(df_agg):,}")
                col3.metric("Total Qty Terjual", f"{int(df_agg['Qty_2022_2025'].sum()):,}")
                
                # 4. Preview Data
                st.markdown("<div class='section-title'>🔍 Preview Data</div>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["📄 Data Mentah", "🧹 Data Hasil Agregasi", "📏 Data Normalisasi"])
                
                with tab1:
                    st.dataframe(df_raw.head(100), use_container_width=True)
                with tab2:
                    st.dataframe(df_agg, use_container_width=True)
                with tab3:
                    st.dataframe(df_scaled, use_container_width=True)

                # 5. TOMBOL KONFIGURASI & PROSES
                st.markdown("---")
                st.markdown("<div class='section-title'>⚙️ Aksi & Konfigurasi Lanjutan</div>", unsafe_allow_html=True)
                st.write("Data telah siap! Klik tombol di bawah untuk menyimpan data dan melanjutkan ke konfigurasi $k$-Means.")
                
                if st.button("🚀 Simpan & Lanjut ke Konfigurasi Clustering", type="primary", use_container_width=True):
                    state.set("upload_done", True)
                    st.success("✅ Preprocessing selesai & data siap dikonfigurasi! Silakan pilih menu **⚙️ Konfigurasi Clustering** pada sidebar di sebelah kiri.")
                    st.balloons()

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            
    else:
        # Pengecekan Aman jika data sudah pernah di-upload sebelumnya
        df_agg = state.get("df_agg")
        if df_agg is not None:
            st.info("ℹ️ Data transaksi sudah diunggah sebelumnya dan tersimpan di memori.")
            
            st.markdown("<div class='section-title'>📊 Ringkasan Data Ter-upload</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("Jumlah Barang Unik", f"{len(df_agg):,}")
            col2.metric("Total Qty Terjual", f"{int(df_agg['Qty_2022_2025'].sum()):,}")
            
            st.dataframe(df_agg, use_container_width=True)
            
            st.markdown("---")
            if st.button("⚙️ Lanjut ke Konfigurasi Clustering", type="primary", use_container_width=True):
                st.success("👉 Silakan buka menu **⚙️ Konfigurasi Clustering** pada sidebar.")
        else:
            st.info("💡 Silakan unggah file transaksi `.xlsx` atau `.csv` untuk memulai.")
