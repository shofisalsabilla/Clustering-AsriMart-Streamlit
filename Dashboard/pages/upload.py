import streamlit as st
import pandas as pd
import utils.clustering as clustering
from utils import state

def show():
    # Inisialisasi state awal
    state.init_state()

    # Header Banner Dark Mode
    st.markdown("""
    <div style='background: #1e1e38; padding: 22px; border-radius: 12px; margin-bottom: 24px;'>
        <h2 style='margin:0; color:white; font-size: 1.8rem;'>📂 Upload & Preprocessing Data</h2>
        <p style='margin:6px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Unggah file Excel, lihat data mentah, lakukan pembersihan otomatis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # 1. UPLOAD FILE EXCEL
    # =========================================================================
    st.markdown("<h4 style='color: #0284c7; margin-bottom: 5px;'>📥 Upload File Excel</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih file .xlsx (contoh: Toko Asri Mart_2022-2025.xlsx)", 
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        try:
            # Membaca File Mentah
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            st.success(f"✅ File berhasil diunggah! **{len(df_raw):,} baris** ditemukan.")
            
            # Validasi Kolom Wajib
            required_columns = {'Nama Barang', 'Qty'}
            if not required_columns.issubset(set(df_raw.columns)):
                st.error(f"❌ Format file salah! File wajib memiliki kolom: **{', '.join(required_columns)}**")
            else:
                # =============================================================
                # 2. PREVIEW DATA MENTAH
                # =============================================================
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #0284c7; margin-bottom: 10px;'>👁️ Preview Data Mentah</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Baris", f"{len(df_raw):,}")
                col2.metric("Total Kolom", f"{len(df_raw.columns)}")
                col3.metric("Null Values", f"{df_raw.isnull().sum().sum():,}")
                
                st.dataframe(df_raw, use_container_width=True, height=280)

                # =============================================================
                # 3. PREPROCESSING OTOMATIS
                # =============================================================
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #0284c7; margin-bottom: 5px;'>🔧 Preprocessing Otomatis</h4>", unsafe_allow_html=True)
                st.write("Sistem akan melakukan:")
                st.markdown("""
                * **Seleksi kolom** &rarr; hanya `Nama Barang` dan `Qty`
                * **Pembersihan** &rarr; hapus nilai NULL, kosong, non-numerik
                * **Agregasi** &rarr; jumlahkan `Qty` per `Nama Barang`
                * **Normalisasi** &rarr; Min-Max Scaler ke rentang [0, 1]
                """)

                # Tombol Eksekusi
                btn_process = st.button("🚀 Jalankan Preprocessing", type="primary", use_container_width=True)
                
                if btn_process or state.get("upload_done"):
                    # Preprocessing Data via clustering.py
                    df_cleaned, df_agg, df_scaled, scaler = clustering.preprocess(df_raw)
                    
                    # Simpan ke State
                    state.set("df_raw", df_raw)
                    state.set("df_cleaned", df_cleaned)
                    state.set("df_agg", df_agg)
                    state.set("df_scaled", df_scaled)
                    state.set("scaler", scaler)
                    state.set("upload_done", True)

                    st.success("✅ Preprocessing selesai!")

                    # =========================================================
                    # 4. DATA BERSIH
                    # =========================================================
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #0284c7; margin-bottom: 0px;'>🧹 Data Bersih</h4>", unsafe_allow_html=True)
                    st.caption(f"**{len(df_cleaned):,} baris** setelah pembersihan")
                    st.dataframe(df_cleaned, use_container_width=True, height=280)

                    # =========================================================
                    # 5. DATA AGREGASI & DATA NORMALISASI (SIDE-BY-SIDE)
                    # =========================================================
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_agg, col_norm = st.columns(2)

                    with col_agg:
                        st.markdown("<h4 style='color: #0284c7; margin-bottom: 10px;'>📊 Data Agregasi</h4>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        c1.metric("Total Barang Unik", f"{len(df_agg):,}")
                        c2.metric("Total Qty", f"{int(df_agg['Qty_2022_2025'].sum()):,}")
                        st.caption(f"**{len(df_agg):,} jenis barang** setelah agregasi")
                        st.dataframe(df_agg, use_container_width=True, height=320)

                    with col_norm:
                        st.markdown("<h4 style='color: #0284c7; margin-bottom: 10px;'>📐 Data Normalisasi</h4>", unsafe_allow_html=True)
                        st.caption("Data setelah **Min-Max Normalisasi** (skala 0–1)")
                        st.markdown("<br>", unsafe_allow_html=True) # penyesuaian jarak vertikal agar sejajar
                        st.dataframe(df_scaled, use_container_width=True, height=320)

                    # =========================================================
                    # 6. BANNER PETUNJUK AKHIR
                    # =========================================================
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("ℹ️ Data siap digunakan. Lanjut ke **⚙️ Konfigurasi Clustering**.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

    else:
        # Tampilan saat file belum dipilih tapi pernah di-upload sebelumnya
        df_agg = state.get("df_agg")
        if df_agg is not None:
            st.info("ℹ️ Data transaksi sudah diunggah sebelumnya dan tersimpan di memori.")
            st.info("💡 Silakan buka menu **⚙️ Konfigurasi Clustering** pada sidebar di sebelah kiri.")
        else:
            st.info("💡 Silakan unggah file transaksi `.xlsx` atau `.csv` untuk memulai.")
