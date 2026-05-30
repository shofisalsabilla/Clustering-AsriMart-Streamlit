import streamlit as st
from utils import state


def show():
    state.init_state()

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0; color:white;'>🏠 Dashboard Utama</h2>
        <p style='margin:4px 0 0; color:#b0c4de; font-size:0.9rem;'>
            Sistem Analisis Clustering K-Means — Toko Asri Mart (2022–2025)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Panel
    upload_done = state.get("upload_done")
    cluster_done = state.get("cluster_done")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        df_raw = state.get("df_raw")
        val = len(df_raw) if df_raw is not None else 0
        st.metric("📦 Total Transaksi", f"{val:,}" if val else "—", help="Jumlah baris data mentah")
    with col2:
        df_agg = state.get("df_agg")
        val = len(df_agg) if df_agg is not None else 0
        st.metric("🏷️ Jenis Barang", f"{val:,}" if val else "—", help="Setelah agregasi per Nama Barang")
    with col3:
        k = state.get("n_clusters") or "—"
        st.metric("⚙️ Jumlah Cluster (k)", k)
    with col4:
        sil = state.get("silhouette_score")
        val = f"{sil:.4f}" if sil is not None else "—"
        st.metric("📈 Silhouette Score", val)

    st.markdown("---")

    # Pipeline Status
    st.markdown("<div class='section-title'>🔄 Status Alur Analisis</div>", unsafe_allow_html=True)

    steps = [
        ("📂 Upload Data", upload_done, "Unggah file Excel (.xlsx)"),
        ("🔧 Preprocessing", upload_done, "Bersihkan data & normalisasi"),
        ("⚙️ Konfigurasi K", cluster_done, "Pilih jumlah cluster & label"),
        ("📊 Clustering", cluster_done, "Jalankan algoritma K-Means"),
        ("📉 Evaluasi", cluster_done, "Elbow method & Silhouette Score"),
    ]

    cols = st.columns(len(steps))
    for col, (label, done, desc) in zip(cols, steps):
        icon = "✅" if done else "⏳"
        color = "#27ae60" if done else "#e67e22"
        bg = "rgba(39,174,96,0.1)" if done else "rgba(230,126,34,0.1)"
        col.markdown(f"""
        <div style='background:{bg}; border:1px solid {color}; border-radius:10px;
                    padding:14px; text-align:center;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='font-weight:600; color:{color}; font-size:0.85rem;'>{label}</div>
            <div style='font-size:0.75rem; color:#7f8c8d; margin-top:4px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Ringkasan hasil
    if cluster_done:
        df_clustered = state.get("df_clustered")
        st.markdown("<div class='section-title'>📊 Ringkasan Hasil Clustering</div>", unsafe_allow_html=True)

        summary = df_clustered.groupby('Kategori').agg(
            Jumlah_Barang=('Nama Barang', 'count'),
            Rata_Qty_Normalisasi=('Qty_2022_2025', 'mean'),
        ).reset_index()

        df_agg = state.get("df_agg")
        merged = df_clustered.merge(df_agg, on='Nama Barang', suffixes=('_norm', '_asli'))
        summary2 = merged.groupby('Kategori').agg(
            Rata_Qty_Asli=('Qty_2022_2025_asli', 'mean'),
        ).reset_index()
        summary = summary.merge(summary2, on='Kategori')

        cols = st.columns(len(summary))
        colors = {"Laris": "#27ae60", "Sedang": "#f39c12", "Kurang Laris": "#e74c3c"}
        for col, (_, row) in zip(cols, summary.iterrows()):
            c = colors.get(row['Kategori'], "#3498db")
            col.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {c};
                        border-radius:12px; padding:18px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:{c};'>{row['Jumlah_Barang']}</div>
                <div style='font-weight:600; color:{c};'>{row['Kategori']}</div>
                <div style='font-size:0.78rem; color:#7f8c8d;'>
                    Rata-rata Qty: {row['Rata_Qty_Asli']:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Mulai dengan **Upload & Preprocessing** di menu sebelah kiri untuk memulai analisis.", icon="ℹ️")

    # Cara penggunaan
    st.markdown("---")
    st.markdown("<div class='section-title'>📖 Panduan Penggunaan</div>", unsafe_allow_html=True)
    with st.expander("Lihat langkah-langkah penggunaan sistem"):
        st.markdown("""
        1. **📂 Upload & Preprocessing** — Unggah file Excel `Toko Asri Mart_2022-2025.xlsx`. Sistem akan membersihkan dan mengagregasi data otomatis.
        2. **⚙️ Konfigurasi Clustering** — Lihat grafik Elbow Method, pilih jumlah cluster *k*, dan atur nama label untuk setiap cluster.
        3. **📊 Hasil Clustering** — Lihat tabel hasil pengelompokan barang beserta jarak Euclidean ke centroid.
        4. **📉 Evaluasi & Visualisasi** — Analisis Silhouette Score, Scatter Plot PCA, dan Bar Chart 10 barang terlaris.
        5. **📋 Laporan** — Unduh hasil clustering dalam format CSV.
        """)
