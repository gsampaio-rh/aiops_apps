import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import csv
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

uploaded_file = "/Users/gsampaio/redhat/ai/ai-music/data/sample_spotify_dataset.csv"
# df = pd.read_csv(uploaded_file)
chunk_size = 50000  # Adjust based on memory capacity

# 🔄 Load data in chunks
chunks = []

for chunk in pd.read_csv(
    uploaded_file,
    sep=",",  # Separator
    quotechar='"',  # Quoting character
    escapechar="\\",  # Escape character
    engine="python",  # Python engine handles complex parsing
    on_bad_lines="skip",  # Skip problematic lines
    chunksize=chunk_size,  # Read in chunks
    quoting=csv.QUOTE_NONE,  # Ignore quotes entirely
    dtype=str,  # Read all columns as strings
    encoding="ISO-8859-1",  # Handle special characters
):
    # 🧹 Clean Column Names
    chunk.columns = chunk.columns.str.replace('"', "").str.strip()

    # 🧹 Clean Cell Values
    chunk = chunk.applymap(lambda x: x.strip('"').strip() if isinstance(x, str) else x)

    chunks.append(chunk)

# 📊 Combine all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)
print("✅ Dataset successfully loaded!")


def build_artist_cooccurrence_fast(df):
    """
    Build an artist co-occurrence matrix where each cell (i, j)
    represents the number of playlists in which artists i and j appear together.
    This version uses a sparse matrix approach for improved efficiency.
    """
    # Drop duplicate artist-playlist pairs to ensure binary presence
    artist_playlist = df[["artistname", "playlistname"]].drop_duplicates()

    # Get unique artists and playlists
    artists = artist_playlist["artistname"].unique()
    playlists = artist_playlist["playlistname"].unique()

    # Create mappings for artists and playlists to their indices
    artist_index = {artist: idx for idx, artist in enumerate(artists)}
    playlist_index = {pl: idx for idx, pl in enumerate(playlists)}

    # Map artist and playlist names to their indices
    rows = artist_playlist["artistname"].map(artist_index).values
    cols = artist_playlist["playlistname"].map(playlist_index).values

    # Create a sparse binary matrix (artists x playlists)
    data = np.ones(len(artist_playlist), dtype=int)
    artist_playlist_matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(artists), len(playlists))
    )

    # Compute the co-occurrence matrix using sparse matrix multiplication
    cooccurrence_sparse = artist_playlist_matrix * artist_playlist_matrix.T

    # Remove self-co-occurrence by setting diagonal to 0
    cooccurrence_sparse.setdiag(0)
    cooccurrence_sparse.eliminate_zeros()

    # Convert the sparse matrix back to a DataFrame (if needed for further processing)
    cooccurrence = pd.DataFrame(
        cooccurrence_sparse.toarray(), index=artists, columns=artists
    )
    return cooccurrence


artist_cooccurrence = build_artist_cooccurrence_fast(df)
artist_cooccurrence

# Normalize the co-occurrence matrix
scaler = StandardScaler()
artist_cooccurrence_scaled = scaler.fit_transform(artist_cooccurrence)
artist_cooccurrence_scaled_df = pd.DataFrame(
    artist_cooccurrence_scaled,
    index=artist_cooccurrence.index,
    columns=artist_cooccurrence.columns,
)

# Dimensionality Reduction using Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
artist_reduced = svd.fit_transform(artist_cooccurrence_scaled_df)
artist_reduced_df = pd.DataFrame(
    artist_reduced,
    index=artist_cooccurrence.index,
    columns=["Component 1", "Component 2"],
)

# Cluster Artists with K-Means
optimal_k = st.slider("Select Number of Clusters", min_value=2, max_value=20, value=10)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(artist_reduced_df)
artist_reduced_df["cluster"] = clusters
