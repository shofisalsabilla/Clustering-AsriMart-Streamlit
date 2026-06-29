import streamlit as st
import pandas as pd
from utils import state, clustering

def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>📂 Upload & Preprocessing Data</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Unggah file Excel, lihat data mentah, lakukan pembersihan otomatis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("<div class='section-title'>📤 Upload File Excel</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file .xlsx (contoh: Toko Asri Mart_2022-2025.xlsx)",
        type=["xlsx"],
        help="File harus memiliki kolom 'Nama Barang' dan 'Qty'"
    )

    if uploaded:
        try:
            df_raw = pd.read_excel(uploaded)
            state.set("df_raw", df_raw)

            # Validasi kolom
            required = {'Nama Barang', 'Qty'}
            if not required.issubset(set(df_raw.columns)):
                st.error(f"❌ File harus memiliki kolom: {required}. Kolom ditemukan: {list(df_raw.columns)}")
                return

            st.success(f"✅ File berhasil diunggah! **{len(df_raw):,} baris** ditemukan.", icon="✅")

            # Preview Data Mentah
            st.markdown("<div class='section-title'>👁️ Preview Data Mentah</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Baris", f"{len(df_raw):,}")
            col2.metric("Total Kolom", len(df_raw.columns))
            col3.metric("Null Values", int(df_raw.isnull().sum().sum()))

            st.dataframe(df_raw.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            return
    elif state.get("df_raw") is None:
        st.warning("⚠️ Belum ada data. Silakan upload file Excel terlebih dahulu.")
        return

    df_raw = state.get("df_raw")

    # Preprocessing
    st.markdown("---")
    st.markdown("<div class='section-title'>🔧 Preprocessing Otomatis</div>", unsafe_allow_html=True)
    st.markdown("""
    Sistem akan melakukan:
    - **Seleksi kolom** → hanya `Nama Barang` dan `Qty`
    - **Pembersihan** → hapus nilai NULL, kosong, non-numerik
    - **Agregasi** → jumlahkan `Qty` per `Nama Barang`
    - **Normalisasi** → Min-Max Scaler ke rentang [0, 1]
    """)

    if st.button("▶️ Jalankan Preprocessing", type="primary", use_container_width=True):
        with st.spinner("Memproses data..."):
            try:
                df_cleaned, df_agg, df_scaled, scaler = clustering.preprocess(df_raw)
                state.set("df_cleaned", df_cleaned)
                state.set("df_agg", df_agg)
                state.set("df_scaled", df_scaled)
                state.set("scaler", scaler)
                state.set("upload_done", True)
                st.success("✅ Preprocessing selesai!", icon="✅")
            except Exception as e:
                st.error(f"❌ Error saat preprocessing: {e}")
                return

    if not state.get("upload_done"):
        return

    df_agg = state.get("df_agg")
    df_scaled = state.get("df_scaled")
    df_cleaned = state.get("df_cleaned")

    # Hasil
    st.markdown("---")

    # 1. Data Bersih
    st.markdown("<div class='section-title'>🧹 Data Bersih</div>", unsafe_allow_html=True)
    st.markdown(f"**{len(df_cleaned):,} baris** setelah pembersihan")
    st.dataframe(df_cleaned.head(50), use_container_width=True)

    st.markdown("---")

    # 2. Data Agregasi & 3. Data Normalisasi (Samping-menyamping)
    col_agregasi, col_normalisasi = st.columns(2)

    with col_agregasi:
        st.markdown("<div class='section-title'>📊 Data Agregasi</div>", unsafe_allow_html=True)
        # Samping-menyampingkan metrik
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.metric("Total Barang Unik", len(df_agg))
        sub_col2.metric("Total Qty", f"{int(df_agg['Qty_2022_2025'].sum()):,}")
        st.markdown(f"**{len(df_agg):,} jenis barang** setelah agregasi")
        st.dataframe(df_agg, use_container_width=True)

    with col_normalisasi:
        st.markdown("<div class='section-title'>📐 Data Normalisasi</div>", unsafe_allow_html=True)
        st.markdown("Data setelah **Min-Max Normalisasi** (skala 0–1)")
        # Penyesuaian spasi agar sejajar dengan bagian atas tabel agregasi
        for _ in range(6):
            st.write("") 
        st.dataframe(df_scaled, use_container_width=True)

    st.markdown("---")
    st.info("✅ Data siap digunakan. Lanjut ke **⚙️ Konfigurasi Clustering**.", icon="ℹ️")
