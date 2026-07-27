import streamlit as st

st.set_page_config(
    page_title="Sistem Clustering K-Means | Toko Asri Mart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# BAGIAN WAJIB UNTUK MENJAGA KONSISTENSI HASIL
# ==========================================
# Inisialisasi variabel global di memori Streamlit
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None # Untuk menyimpan data mentah awal
if 'df_scaled' not in st.session_state:
    st.session_state['df_scaled'] = None # Untuk menyimpan data yang sudah dinormalisasi
if 'kmeans_model' not in st.session_state:
    st.session_state['kmeans_model'] = None # Untuk memuat model dari Notebook (.pkl)
if 'scaler_model' not in st.session_state:
    st.session_state['scaler_model'] = None # Untuk memuat scaler dari Notebook (.pkl)
if 'df_result' not in st.session_state:
    st.session_state['df_result'] = None # Untuk menyimpan hasil akhir clustering
# ==========================================

# Sembunyikan navigasi otomatis Streamlit
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ... [BAGIAN CUSTOM CSS ANDA TETAP SAMA DI SINI, SAYA SKIP AGAR RINGKAS] ...

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:2.5rem;'>🛒</div>
        <div style='font-weight:700; font-size:1.1rem; color:#4fc3f7;'>Toko Asri Mart</div>
        <div style='font-size:0.78rem; color:#7f8c8d;'>Sistem Analisis Clustering</div>
    </div>
    <hr style='border-color:#2a4a7f; margin-bottom:16px;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "📌 Menu Navigasi",
        options=[
            "🏠 Dashboard",
            "📂 Upload & Preprocessing",
            "⚙️ Konfigurasi Clustering",
            "📊 Hasil Clustering",
            "📉 Evaluasi & Visualisasi",
        ],
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:#2a4a7f; margin-top:20px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a5568; text-align:center; padding-top:8px;'>
        K-Means Clustering v1.0<br>Skripsi — Analisis Qty
    </div>
    """, unsafe_allow_html=True)

# Route ke halaman
page_name = menu.split(" ", 1)[1].strip()

if page_name == "Dashboard":
    from pages import dashboard
    dashboard.show()
elif page_name == "Upload & Preprocessing":
    from pages import upload
    upload.show()
elif page_name == "Konfigurasi Clustering":
    from pages import konfigurasi
    konfigurasi.show()
elif page_name == "Hasil Clustering":
    from pages import hasil
    hasil.show()
elif page_name == "Evaluasi & Visualisasi":
    from pages import evaluasi
    evaluasi.show()
