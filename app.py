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

# App Configurations
st.set_page_config(page_title="Spotify Recommender", layout="wide")
st.title("🎵 Spotify Playlist & Artist Recommender")

# Upload Dataset
uploaded_file = st.file_uploader("📁 Upload a Spotify Playlist Dataset (CSV)", type=["csv"])
if uploaded_file:
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
    st.success("✅ Dataset successfully loaded!")
    st.dataframe(df.head())

    # Basic Stats
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", df['user_id'].nunique())
    col2.metric("Artists", df['artistname'].nunique())
    col3.metric("Playlists", df['playlistname'].nunique())
    col4.metric("Tracks", df['trackname'].nunique())

    # Top Artists Visualization
    st.subheader("🎤 Top Artists")
    top_artists = df["artistname"].value_counts().head(10)
    fig = px.bar(top_artists, x=top_artists.index, y=top_artists.values, title="Top 10 Most Popular Artists")
    st.plotly_chart(fig)

    # Collaborative Filtering Preparation
    st.subheader("🔍 Collaborative Filtering Recommendation")
    user_artist_matrix = df.pivot_table(
        index="user_id", columns="artistname", aggfunc="size", fill_value=0
    )
    user_playlist_matrix = df.pivot_table(
        index="user_id", columns="playlistname", aggfunc="size", fill_value=0
    )
    user_artist_sparse = csr_matrix(user_artist_matrix.values)
    user_playlist_sparse = csr_matrix(user_playlist_matrix.values)
    user_artist_similarity = cosine_similarity(user_artist_sparse)
    similarity_df = pd.DataFrame(
        user_artist_similarity,
        index=user_artist_matrix.index,
        columns=user_artist_matrix.index,
    )

    # User Summary
    def user_summary(user_id):
        user_playlists = df[df["user_id"] == user_id]["playlistname"].value_counts()
        user_artists = df[df["user_id"] == user_id]["artistname"].value_counts()

        with st.expander(f"🎧 Playlists for User {user_id}"):
            for playlist, count in user_playlists.items():
                st.write(f"📁 {playlist} ({count} tracks)")

        with st.expander(f"🎤 Favorite Artists for User {user_id}"):
            for artist, count in user_artists.items():
                st.write(f"🎵 {artist} ({count} tracks)")

    # User Input for Recommendation
    st.subheader("🎧 Get Personalized Artist Recommendations")
    selected_user = st.selectbox("Select a User ID", df["user_id"].unique())
    if selected_user:
        user_summary(selected_user)

    def recommend_artists(user_id, top_n=5):
        if user_id not in similarity_df.index:
            return []
        similar_users = (
            similarity_df[user_id]
            .sort_values(ascending=False)
            .iloc[1 : top_n + 1]
            .index
        )
        user_artists = set(
            user_artist_matrix.loc[user_id][user_artist_matrix.loc[user_id] > 0].index
        )
        recommendations = []
        for similar_user in similar_users:
            similar_artists = set(
                user_artist_matrix.loc[similar_user][
                    user_artist_matrix.loc[similar_user] > 0
                ].index
            )
            new_artists = similar_artists - user_artists
            for artist in new_artists:
                recommendations.append(f"{artist} - Matched with {similar_user}")
            if len(recommendations) >= top_n:
                break
        return recommendations[:top_n]

    if st.button("🔍 Recommend Artists"):
        with st.spinner("🔄 Generating recommendations..."):
            recommendations = recommend_artists(selected_user, top_n=5)
            st.success("✅ Done!")
            st.write("🎤 Recommended Artists:", recommendations if recommendations else "No recommendations found.")

    # Network Graph Visualization
    def create_network_graph(user_id, top_n=5):
        G = nx.Graph()
        similar_users = (
            similarity_df[user_id].sort_values(ascending=False).head(top_n).index
        )
        recommended_artists = recommend_artists(user_id, top_n)

        for similar_user in similar_users:
            G.add_edge(
                user_id, similar_user, weight=similarity_df.at[user_id, similar_user]
            )
            for artist in recommended_artists:
                G.add_edge(similar_user, artist, weight=0.5)
                G.nodes[artist]["color"] = "red"

        pos = nx.spring_layout(G)
        node_colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes()]
        plt.figure(figsize=(10, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=2000,
            font_size=10,
        )
        st.pyplot(plt)

    if st.button("🔍 Show Recommendation Graph"):
        create_network_graph(selected_user, top_n=5)

    # Heatmap of Similarity
    st.subheader("🌐 How These Matches Were Made")
    # Select top 10 most similar users
    similar_users = (
        similarity_df[selected_user].sort_values(ascending=False).head(10).index
    )
    heatmap_data = similarity_df.loc[similar_users, similar_users]

    # Truncate usernames for better readability
    truncated_labels = {
        user: user[:6] + "..." if len(user) > 6 else user for user in similar_users
    }
    heatmap_data.index = [truncated_labels[user] for user in heatmap_data.index]
    heatmap_data.columns = [truncated_labels[user] for user in heatmap_data.columns]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        linewidths=0.5,
        annot=True,  # ✅ Show similarity scores
        fmt=".2f",  # ✅ Format to 2 decimal places
        annot_kws={"size": 5},  # ✅ Font size for annotations
    )

    st.pyplot(fig)

    # Create Artist-User Matrix
    artist_user_df = (
        df.groupby(["artistname", "user_id"]).size().reset_index(name="track_count")
    )
    artist_user_matrix = artist_user_df.pivot_table(
        index="artistname", columns="user_id", values="track_count", fill_value=0
    )

    # Normalize Data
    scaler = StandardScaler()
    artist_user_matrix_scaled = scaler.fit_transform(artist_user_matrix)
    artist_user_matrix_scaled_df = pd.DataFrame(
        artist_user_matrix_scaled,
        index=artist_user_matrix.index,
        columns=artist_user_matrix.columns,
    )

    # Dimensionality Reduction with Truncated SVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    artist_user_pca = svd.fit_transform(artist_user_matrix_scaled_df)
    artist_user_pca_df = pd.DataFrame(
        artist_user_pca, index=artist_user_matrix.index, columns=["PCA1", "PCA2"]
    )

    # Clustering with K-Means
    optimal_k = 10
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(artist_user_pca_df)
    artist_user_pca_df["cluster"] = user_clusters

    # Silhouette Score
    sil_score = silhouette_score(
        artist_user_pca_df[["PCA1", "PCA2"]], artist_user_pca_df["cluster"]
    )
    st.subheader(f"⭐ Silhouette Score: {sil_score:.2f}")

    # User Cluster Visualization
    st.subheader("🎨 User Clusters Based on Artist Preferences")
    fig = px.scatter(
        artist_user_pca_df,
        x="PCA1",
        y="PCA2",
        color=artist_user_pca_df["cluster"].astype(str),
        title="User Clusters",
        labels={"cluster": "Cluster"},
    )
    st.plotly_chart(fig)

    selected_user = st.selectbox(
        "Select a User ID for Cluster-Based Recommendations", df["artistname"].unique()
    )
    if selected_user:
        if selected_user in artist_user_pca_df.index:
            user_cluster = artist_user_pca_df.loc[selected_user, "cluster"]
            similar_users = artist_user_pca_df[artist_user_pca_df["cluster"] == user_cluster].index.tolist()
            st.write(f"👥 Users in the same cluster as {selected_user}: {similar_users}")
        else:
            st.error(f"User {selected_user} is not found in the clustering results. Please select another user.")
