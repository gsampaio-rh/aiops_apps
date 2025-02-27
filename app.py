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


def get_similar_users_and_recommendations(
    user_id, user_artist_matrix, similarity_df, top_n_users=10
):
    """
    Returns the top similar users and recommended artists for a given user.

    Parameters:
        user_id (str): The selected user ID.
        user_artist_matrix (DataFrame): User-Artist interaction matrix.
        similarity_df (DataFrame): DataFrame of user similarity scores.
        top_n_users (int): Number of similar users to consider.

    Returns:
        similar_users (Index): Similar users (excluding the selected user).
        recommended_artists (list): Artists liked by similar users but not by the selected user.
    """
    if user_id not in similarity_df.index:
        return [], []

    similar_users = (
        similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1 : top_n_users + 1]
        .index
    )

    selected_user_artists = set(
        user_artist_matrix.loc[user_id][user_artist_matrix.loc[user_id] > 0].index
    )

    recommended_artists = set()
    for su in similar_users:
        su_artists = set(
            user_artist_matrix.loc[su][user_artist_matrix.loc[su] > 0].index
        )
        recommended_artists.update(su_artists - selected_user_artists)

    return similar_users, list(recommended_artists)


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
    top_artists = df["artistname"].value_counts().head(20)
    fig = px.bar(top_artists, x=top_artists.index, y=top_artists.values, title="Top 10 Most Popular Artists")
    st.plotly_chart(fig)

    # User Summary
    def user_summary(user_id):
        user_playlists = df[df["user_id"] == user_id]["playlistname"].value_counts()
        user_artists = df[df["user_id"] == user_id]["artistname"].value_counts()

        st.subheader(f"👤 User Profile: {user_id}")

        # 🎵 Playlists Section with Dropdown
        with st.expander("📂 Playlists"):
            if user_playlists.empty:
                st.info("No playlists found for this user.")
            else:
                cols = st.columns(min(8, len(user_playlists)))  # Responsive grid
                for idx, (playlist, count) in enumerate(user_playlists.items()):
                    with cols[idx % len(cols)]:
                        st.markdown(
                            f"""
                            <div style="
                                border-radius: 10px;
                                padding: 10px;
                                margin: 10px;
                                background: linear-gradient(135deg, #ff758c, #ff7eb3);
                                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                                text-align: center;
                                font-size: 14px;
                                transition: transform 0.2s ease-in-out;
                            ">
                            <h4 style="color: white; margin: 5px;">📁 {playlist}</h4>
                            <p style="color: white; margin: 2px;">{count} tracks</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # 🎤 Favorite Artists Section with Dropdown
        with st.expander("🎤 Favorite Artists"):
            if user_artists.empty:
                st.info("No favorite artists found for this user.")
            else:
                cols = st.columns(min(8, len(user_artists)))  # More compact layout
                for idx, (artist, count) in enumerate(user_artists.items()):
                    with cols[idx % len(cols)]:
                        st.markdown(
                            f"""
                            <div style="
                                border-radius: 10px;
                                padding: 10px;
                                margin: 10px;
                                background: linear-gradient(135deg, #a18cd1, #fbc2eb);
                                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                                text-align: center;
                                font-size: 14px;
                                transition: transform 0.2s ease-in-out;
                            ">
                            <h4 style="color: white; margin: 5px;">🎵 {artist}</h4>
                            <p style="color: white; margin: 2px;">{count} tracks</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # User Input for Recommendation
    st.subheader("🎧 Get Personalized Artist Recommendations")
    selected_user = st.selectbox("Select a User ID", df["user_id"].unique())
    if selected_user:
        user_summary(selected_user)

    # Collaborative Filtering Preparation
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

    # 📌 Heatmap of Similarity
    st.subheader("🌐 Matchmaking Network")
    with st.expander("🔥 View User Similarity Heatmap"):
        # Select top 10 most similar users
        similar_users = (
            similarity_df[selected_user].sort_values(ascending=False).head(10).index
        )
        heatmap_data = similarity_df.loc[similar_users, similar_users]

        # 📌 Improve Label Readability
        max_label_length = 8  # Adjust for better readability
        truncated_labels = {
            user: user[:max_label_length] + "..." if len(user) > max_label_length else user
            for user in similar_users
        }
        heatmap_data.index = [truncated_labels[user] for user in heatmap_data.index]
        heatmap_data.columns = [truncated_labels[user] for user in heatmap_data.columns]

        # 🎨 Improved Heatmap Design
        fig, ax = plt.subplots(figsize=(9, 6))  # Adjust for readability
        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",  # Better contrast
            linewidths=0.7,  # Slimmer grid lines for cleaner look
            annot=True,  # ✅ Show similarity scores
            fmt=".2f",  # ✅ Format to 2 decimal places
            annot_kws={"size": 7},  # ✅ Better font styling
            cbar_kws={"shrink": 0.75},  # ✅ Shrink color bar for aesthetics
        )

        # 🔄 Dynamic Label Rotation Based on Length
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        # 🖼️ Display the heatmap
        st.pyplot(fig)

    with st.expander("🤝 View User-Artist Interaction Grid - Collaborative Filtering Recommendation"):
        if selected_user not in similarity_df.index:
            st.warning("⚠️ Selected user not found in similarity matrix.")
        else:
            # 🎛️ User Control for Top N Users & Artists
            col1, col2 = st.columns(2)
            top_n_users = col1.slider("🔢 Number of Similar Users", 3, 20, 10)
            top_n_artists = col2.slider("🎼 Number of Artists", 3, 50, 10)

            # Use the helper function to compute similar users and recommended artists
            similar_users, recommended_artists = get_similar_users_and_recommendations(
                selected_user, user_artist_matrix, similarity_df, top_n_users=top_n_users
            )
            # Filter the user-artist matrix to include the selected user and their top similar users
            filtered_matrix = user_artist_matrix.loc[[selected_user] + list(similar_users)]

            # Determine the set of artists the selected user has interacted with
            selected_user_artists = set(
                user_artist_matrix.loc[selected_user][user_artist_matrix.loc[selected_user] > 0].index
            )

            # Order columns: first, the selected user's liked artists; then, recommended artists.
            selected_artists_in_matrix = [artist for artist in filtered_matrix.columns if artist in selected_user_artists]
            recommended_artists_in_matrix = [artist for artist in filtered_matrix.columns if artist in recommended_artists]
            # Combine them, ensuring no duplicates and limiting to top_n_artists
            ordered_artists = selected_artists_in_matrix + [
                artist for artist in recommended_artists_in_matrix if artist not in selected_artists_in_matrix
            ]
            ordered_artists = ordered_artists[:top_n_artists]
            filtered_matrix = filtered_matrix[ordered_artists]

            # 🎭 Truncate long usernames & artist names for readability
            max_label_length = 10  # Limit to 10 characters
            truncated_users = {
                user: (user[:max_label_length] + "..." if len(user) > max_label_length else user)
                for user in filtered_matrix.index
            }
            truncated_artists = {
                artist: (artist[:max_label_length] + "..." if len(artist) > max_label_length else artist)
                for artist in filtered_matrix.columns
            }
            filtered_matrix.index = [truncated_users[user] for user in filtered_matrix.index]
            filtered_matrix.columns = [truncated_artists[artist] for artist in filtered_matrix.columns]

            # 🎭 Generate the visual grid matrix
            fig, ax = plt.subplots(figsize=(9, 5))
            matrix_data = filtered_matrix.to_numpy()

            # Replace numerical values with icons: "✔️" for interaction, "❌" for none
            # We'll add a 💡 for new recommended artists
            icon_matrix = np.full(matrix_data.shape, "❌", dtype=object)

            # Mark interactions (✔️)
            icon_matrix[matrix_data > 0] = "✔️"

            # Mark new recommended artists with 💡 only for the relevant cells
            for row_idx, user in enumerate(filtered_matrix.index):
                for col_idx, artist in enumerate(filtered_matrix.columns):
                    # Check if this artist is recommended but not yet interacted with
                    if (
                        artist in recommended_artists_in_matrix
                        and artist not in selected_user_artists
                    ):
                        if matrix_data[row_idx, col_idx] == 0:  # No interaction by this user
                            icon_matrix[row_idx, col_idx] = "💡"

            # Convert icon_matrix to DataFrame
            grid_df = pd.DataFrame(icon_matrix, index=filtered_matrix.index, columns=filtered_matrix.columns)

            # Display the grid with styled icons
            st.dataframe(grid_df.style.set_properties(**{"text-align": "center"}))

    with st.expander("🔍 View Artist Recommendations"):
        if selected_user not in similarity_df.index:
            st.warning("⚠️ Selected user not found in similarity matrix.")
        else:
            # 🎛️ User Control for Top N Users & Artists with unique keys
            col1, col2 = st.columns(2)
            top_n_users = col1.slider(
                "🔢 Number of Similar Users", 3, 20, 10, key="recommendations_user_slider"
            )
            top_n_artists = col2.slider(
                "🎼 Number of Artists",
                3,
                50,
                10,
                key="recommendations_artists_slider",
            )

            # Use the helper function to compute similar users and recommended artists
            similar_users, recommended_artists = get_similar_users_and_recommendations(
                selected_user, user_artist_matrix, similarity_df, top_n_users=top_n_users
            )

            if not recommended_artists:
                st.info("⚠️ No recommendations found. The selected user may have already interacted with all artists.")
            else:
                st.subheader(f"Recommended Artists for {selected_user} (Based on Similar Users)")

                # Display each recommended artist with a 💡 icon and the suggesting user
                for idx, artist in enumerate(recommended_artists[:top_n_artists], start=1):
                    st.markdown(
                        f"**{idx}.** 💡 {artist} (Suggested by: {', '.join(similar_users)})"
                    )

                # If desired, add a note for recommended artists
                st.info(f"Above are the top {top_n_artists} recommended artists based on similar users.")

    
    # # Network Graph Visualization
    # def create_network_graph(user_id, top_n=5):
    #     G = nx.Graph()
    #     similar_users = (
    #         similarity_df[user_id].sort_values(ascending=False).head(top_n).index
    #     )
    #     recommended_artists = recommend_artists(user_id, top_n)

    #     for similar_user in similar_users:
    #         G.add_edge(
    #             user_id, similar_user, weight=similarity_df.at[user_id, similar_user]
    #         )
    #         for artist in recommended_artists:
    #             G.add_edge(similar_user, artist, weight=0.5)
    #             G.nodes[artist]["color"] = "red"

    #     pos = nx.spring_layout(G)
    #     node_colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes()]
    #     plt.figure(figsize=(10, 6))
    #     nx.draw(
    #         G,
    #         pos,
    #         with_labels=True,
    #         node_color=node_colors,
    #         edge_color="gray",
    #         node_size=2000,
    #         font_size=10,
    #     )
    #     st.pyplot(plt)

    # if st.button("🔍 Show Recommendation Graph"):
    #     create_network_graph(selected_user, top_n=5)

    # # Create Artist-User Matrix
    # artist_user_df = (
    #     df.groupby(["artistname", "user_id"]).size().reset_index(name="track_count")
    # )
    # artist_user_matrix = artist_user_df.pivot_table(
    #     index="artistname", columns="user_id", values="track_count", fill_value=0
    # )

    # # Normalize Data
    # scaler = StandardScaler()
    # artist_user_matrix_scaled = scaler.fit_transform(artist_user_matrix)
    # artist_user_matrix_scaled_df = pd.DataFrame(
    #     artist_user_matrix_scaled,
    #     index=artist_user_matrix.index,
    #     columns=artist_user_matrix.columns,
    # )

    # # Dimensionality Reduction with Truncated SVD
    # svd = TruncatedSVD(n_components=2, random_state=42)
    # artist_user_pca = svd.fit_transform(artist_user_matrix_scaled_df)
    # artist_user_pca_df = pd.DataFrame(
    #     artist_user_pca, index=artist_user_matrix.index, columns=["PCA1", "PCA2"]
    # )

    # # Clustering with K-Means
    # optimal_k = 10
    # kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    # user_clusters = kmeans.fit_predict(artist_user_pca_df)
    # artist_user_pca_df["cluster"] = user_clusters

    # # Silhouette Score
    # sil_score = silhouette_score(
    #     artist_user_pca_df[["PCA1", "PCA2"]], artist_user_pca_df["cluster"]
    # )
    # st.subheader(f"⭐ Silhouette Score: {sil_score:.2f}")

    # # User Cluster Visualization
    # st.subheader("🎨 User Clusters Based on Artist Preferences")
    # fig = px.scatter(
    #     artist_user_pca_df,
    #     x="PCA1",
    #     y="PCA2",
    #     color=artist_user_pca_df["cluster"].astype(str),
    #     title="Artist Clusters",
    #     labels={"cluster": "Cluster"},
    # )
    # st.plotly_chart(fig)

    # selected_user = st.selectbox(
    #     "Select an Artist for Cluster-Based Recommendations", df["artistname"].unique()
    # )
    # if selected_user:
    #     if selected_user in artist_user_pca_df.index:
    #         user_cluster = artist_user_pca_df.loc[selected_user, "cluster"]
    #         similar_users = artist_user_pca_df[artist_user_pca_df["cluster"] == user_cluster].index.tolist()
    #         st.write(f"👥 Users in the same cluster as {selected_user}: {similar_users}")
    #     else:
    #         st.error(f"User {selected_user} is not found in the clustering results. Please select another user.")
