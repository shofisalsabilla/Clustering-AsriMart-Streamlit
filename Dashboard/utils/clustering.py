import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocess(df_raw):
    """
    Membersihkan data mentah, mengagregasi Qty per Nama Barang,
    dan melakukan normalisasi MinMax.
    """
    # 1. Seleksi Kolom
    df_cleaned = df_raw[['Nama Barang', 'Qty']].copy()
    
    # 2. Cleaning Nilai (NULL, Kosong, String 0, Simbol)
    df_cleaned['Qty'] = (
        df_cleaned['Qty']
        .astype(str)
        .str.strip()
        .replace(['', ' ', 'NULL', 'NaN', 'nan', '-', '0'], np.nan)
    )
    df_cleaned['Qty'] = pd.to_numeric(df_cleaned['Qty'], errors='coerce')
    df_cleaned.dropna(subset=['Qty'], inplace=True)
    
    # 3. Agregasi Qty per Nama Barang (Disesuaikan menjadi Qty_2022_2025)
    df_agg = df_cleaned.groupby('Nama Barang')['Qty'].sum().reset_index()
    df_agg.rename(columns={'Qty': 'Qty_2022_2025'}, inplace=True)
    df_agg = df_agg.sort_values(by='Qty_2022_2025', ascending=False).reset_index(drop=True)
    
    # 4. Normalisasi dengan MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = df_agg.copy()
    df_scaled['Qty_2022_2025_Scaled'] = scaler.fit_transform(df_agg[['Qty_2022_2025']])
    
    return df_cleaned, df_agg, df_scaled, scaler


def run_kmeans(df_scaled, n_clusters=3):
    """
    Menjalankan K-Means Clustering dan menghitung Silhouette Score.
    """
    X = df_scaled[['Qty_2022_2025_Scaled']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    
    df_result = df_scaled.copy()
    df_result['Cluster'] = labels
    
    return df_result, kmeans, score
