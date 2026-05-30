import streamlit as st

st.set_page_config(
    page_title="Sistem Clustering K-Means | Toko Asri Mart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p {
        color: #a0aec0 !important;
    }

    /* Card metric */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #2a4a7f;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f3460, #533483);
        padding: 20px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        color: white;
    }

    /* Section header */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #4fc3f7;
        border-left: 4px solid #4fc3f7;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }

    /* Nav item active */
    .nav-active {
        background: rgba(79, 195, 247, 0.15);
        border-radius: 8px;
        padding: 6px 12px;
    }

    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

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
