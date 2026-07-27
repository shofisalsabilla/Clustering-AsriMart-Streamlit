import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocess(df_raw):
    """
    Fungsi untuk membersihkan data, melakukan agregasi, dan normalisasi.
    """
    # 1. Seleksi Kolom
    df_cleaned = df_raw[['Nama Barang', 'Qty']].copy()
    
    # 2. Pembersihan Nilai (String, NULL, 0, Karakter Aneh)
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
    df_agg.rename(columns={'Qty': 'Total_Qty'}, inplace=True)
    
    # Sortir berdasarkan Total_Qty tertinggi
    df_agg = df_agg.sort_values(by='Total_Qty', ascending=False).reset_index(drop=True)
    
    # 4. Normalisasi dengan MinMaxScaler (0 - 1)
    scaler = MinMaxScaler()
    df_scaled = df_agg.copy()
    df_scaled['Total_Qty_Scaled'] = scaler.fit_transform(df_agg[['Total_Qty']])
    
    return df_cleaned, df_agg, df_scaled, scaler


def run_kmeans(df_scaled, n_clusters=3):
    """
    Fungsi untuk menjalankan K-Means Clustering pada data yang sudah dinormalisasi.
    """
    X = df_scaled[['Total_Qty_Scaled']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Hitung Silhouette Score jika sampel mencukupi
    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    
    # Gabungkan hasil cluster ke dataframe
    df_result = df_scaled.copy()
    df_result['Cluster'] = labels
    
    return df_result, kmeans, score
