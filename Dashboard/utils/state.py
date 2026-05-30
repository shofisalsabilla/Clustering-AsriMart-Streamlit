import streamlit as st

def init_state():
    """Initialize all session state variables."""
    defaults = {
        "df_raw": None,
        "df_cleaned": None,
        "df_agg": None,
        "df_scaled": None,
        "df_clustered": None,
        "df_distances": None,
        "kmeans_model": None,
        "scaler": None,
        "wcss": None,
        "silhouette_score": None,
        "n_clusters": 3,
        "cluster_labels": {0: "Kurang Laris", 1: "Laris", 2: "Sedang"},
        "upload_done": False,
        "cluster_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def get(key):
    return st.session_state.get(key)

def set(key, value):
    st.session_state[key] = value
