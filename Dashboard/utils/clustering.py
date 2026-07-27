import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocess(df_raw):
    """
    Membersihkan data mentah, mengagregasi Qty per Nama Barang,
    dan melakukan normalisasi MinMax (0 - 1).
    """
    # 1. Seleksi Kolom Utama
    df_cleaned = df_raw[['Nama Barang', 'Qty']].copy()
    
    # 2. Cleaning Nilai (Pembersihan string kosong, NULL, "-", dan "0")
    df_cleaned['Qty'] = (
        df_cleaned['Qty']
        .astype(str)
        .str.strip()
        .replace(['', ' ', 'NULL', 'NaN', 'nan', '-', '0'], np.nan)
    )
    df_cleaned['Qty'] = pd.to_numeric(df_cleaned['Qty'], errors='coerce')
    df_cleaned.dropna(subset=['Qty'], inplace=True)
    
    # 3. Agregasi Total Qty per Nama Barang
    df_agg = df_cleaned.groupby('Nama Barang')['Qty'].sum().reset_index()
    df_agg.rename(columns={'Qty': 'Qty_2022_2025'}, inplace=True)
    df_agg = df_agg.sort_values(by='Qty_2022_2025', ascending=False).reset_index(drop=True)
    
    # 4. Normalisasi dengan MinMaxScaler (0 - 1)
    scaler = MinMaxScaler()
    df_scaled = df_agg.copy()
    df_scaled['Qty_2022_2025_Scaled'] = scaler.fit_transform(df_agg[['Qty_2022_2025']])
    
    return df_cleaned, df_agg, df_scaled, scaler


def compute_elbow(df_scaled, k_max=10):
    """
    Menghitung WCSS (Inertia) dari k=1 sampai k_max untuk grafik Elbow Method.
    """
    X = df_scaled[['Qty_2022_2025_Scaled']].values
    wcss = []
    
    max_k = min(k_max, len(X))
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(
            n_clusters=k, 
            init='random',      # Disesuaikan agar sama dengan notebook
            random_state=42,    # Samakan nilai seed ini dengan notebook
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    return wcss


def run_kmeans(df_scaled, n_clusters=3, label_map=None, random_state=42):
    """
    Menjalankan K-Means Clustering dengan inisialisasi acak, 
    menghitung Silhouette Score, dan jarak Euclidean ke centroid.
    
    Returns: (kmeans_model, df_clustered, df_dist, silhouette_score, centroids)
    """
    X = df_scaled[['Qty_2022_2025_Scaled']].values
    
    # Menggunakan init='random' seperti di notebook Anda
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init='random',          # Inisialisasi centroid secara acak
        random_state=random_state, # Seed acak
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Hitung Silhouette Score
    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    
    # Dataframe Hasil Clustering
    df_clustered = df_scaled.copy()
    df_clustered['Cluster'] = labels
    
    if label_map:
        df_clustered['Kategori'] = df_clustered['Cluster'].map(label_map)
        
    # Hitung Jarak Euclidean ke Masing-Masing Centroid
    distances = kmeans.transform(X)
    df_dist = pd.DataFrame(
        distances, 
        columns=[f"Jarak_Centroid_{i+1}" for i in range(n_clusters)]
    )
    df_dist.insert(0, 'Nama Barang', df_scaled['Nama Barang'])
    
    return kmeans, df_clustered, df_dist, score, centroids
