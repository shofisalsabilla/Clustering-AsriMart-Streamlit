import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


def preprocess(df_raw):
    """Clean and aggregate raw data."""
    data = df_raw[['Nama Barang', 'Qty']].copy()

    # Bersihkan Qty
    data['Qty'] = (
        data['Qty']
        .astype(str)
        .str.strip()
        .replace(['', ' ', 'NULL', 'NaN', 'nan', '-', '0'], np.nan)
    )
    data['Qty'] = pd.to_numeric(data['Qty'], errors='coerce')
    data = data.dropna(subset=['Qty'])

    # Isi nama barang kosong
    if data['Nama Barang'].isnull().any():
        data['Nama Barang'] = data['Nama Barang'].fillna(
            data['Nama Barang'].mode()[0]
        )
    data = data.dropna().reset_index(drop=True)

    # Agregasi
    df_agg = data.groupby('Nama Barang')['Qty'].sum().reset_index()
    df_agg.rename(columns={'Qty': 'Qty_2022_2025'}, inplace=True)

    # Normalisasi
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_agg[['Qty_2022_2025']])
    df_scaled = pd.DataFrame(scaled, columns=['Qty_2022_2025'])
    df_scaled.insert(0, 'Nama Barang', df_agg['Nama Barang'].values)

    return data, df_agg, df_scaled, scaler


def compute_elbow(df_scaled, k_max=10):
    """Return WCSS list for k=1..k_max."""
    X = df_scaled[['Qty_2022_2025']]
    wcss = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)
    return wcss


def run_kmeans(df_scaled, n_clusters, label_map):
    """Run K-Means and return model, clustered df, distances df, silhouette."""
    X = df_scaled[['Qty_2022_2025']]

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    clusters = model.predict(X)

    # Distances
    distances = cdist(X, model.cluster_centers_, metric='euclidean')
    df_dist = pd.DataFrame(
        distances,
        columns=[f'Jarak ke Centroid {i+1}' for i in range(n_clusters)]
    )
    df_dist.insert(0, 'Nama Barang', df_scaled['Nama Barang'].values)

    # Clustered df
    df_clustered = df_scaled.copy()
    df_clustered['Cluster'] = clusters

    # Map label — dinamis jika jumlah cluster berubah
    dynamic_map = {}
    cluster_means = (
        df_clustered.groupby('Cluster')['Qty_2022_2025']
        .mean()
        .sort_values()
    )
    sorted_clusters = cluster_means.index.tolist()

    default_names = ["Kurang Laris", "Sedang", "Laris"]
    for rank, cid in enumerate(sorted_clusters):
        if cid in label_map:
            dynamic_map[cid] = label_map[cid]
        elif rank < len(default_names):
            dynamic_map[cid] = default_names[rank]
        else:
            dynamic_map[cid] = f"Cluster {cid}"

    df_clustered['Kategori'] = df_clustered['Cluster'].map(dynamic_map)

    sil = silhouette_score(X, clusters)

    return model, df_clustered, df_dist, sil, dynamic_map
