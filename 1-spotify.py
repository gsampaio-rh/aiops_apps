import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

# ---- TABBED LAYOUT ----
tabs = st.tabs(
    [
        "Spotify Dataset",
        "User Profile",
        "Artists Clusters",
    ]
)

# Empty DataFrame with no columns or index
df = pd.DataFrame()

# ---- DATASET TAB ----
with tabs[0]:
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


# ---- 🎯 Reduce User Similarity Matrix to 2D ----
@st.cache_data
def compute_user_embeddings(similarity_df, method="TSNE"):
    """
    Reduces the user similarity matrix to 2D using PCA or TSNE.
    """
    user_ids = similarity_df.index

    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
    else:  # Default to TSNE for better separation
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    user_embeddings = reducer.fit_transform(similarity_df)

    return pd.DataFrame(user_embeddings, columns=["X", "Y"], index=user_ids)


# ---- 🎨 Generate 2D User Plot ----
def plot_user_scatter(user_embeddings):
    """
    Creates an interactive scatter plot of users in 2D space.
    """
    fig = px.scatter(
        user_embeddings,
        x="X",
        y="Y",
        text=user_embeddings.index,  # User IDs as labels
        title="2D User Similarity Projection",
        labels={"X": "Dimension 1", "Y": "Dimension 2"},
        hover_data={"X": False, "Y": False},  # Hide redundant hover info
    )

    fig.update_traces(
        marker=dict(size=10, color="blue", opacity=0.7)
    )  # Adjust dot size & color

    st.plotly_chart(fig)

def create_user_artist_shared_network(
    similarity_df, user_artist_matrix, selected_user, top_n_users=7, top_n_artists=5
):
    """
    Creates a force-directed graph where:
    - Users (🔵) connect via shared artists (🟢)
    - Recommended artists (🟡) connect to the user providing the recommendation
    """
    G = nx.Graph()

    # 🎯 Add Main User
    G.add_node(selected_user, size=30, color="red", label=f"🎯 {selected_user}")

    # 👥 Find Similar Users
    similar_users = (
        similarity_df[selected_user]
        .sort_values(ascending=False)
        .head(top_n_users)
        .index
    )

    for user in similar_users:
        if user == selected_user:
            continue  # Don't overwrite the selected user's node
        G.add_node(user, size=20, color="blue", label=f"👥 {user}")

    # 🎵 Artists Liked by Main User
    main_user_artists = set(
        user_artist_matrix.loc[selected_user][
            user_artist_matrix.loc[selected_user] > 0
        ].index
    )

    # 🔗 Shared Artists Between Users
    artist_connections = {}  # Store which artists connect which users
    for user in similar_users:
        user_artists = set(
            user_artist_matrix.loc[user][user_artist_matrix.loc[user] > 0].index
        )
        shared_artists = main_user_artists.intersection(
            user_artists
        )  # Find common artists

        for artist in shared_artists:
            if artist not in artist_connections:
                artist_connections[artist] = set()
            artist_connections[artist].update(
                [selected_user, user]
            )  # Store which users like this artist

    # 🎵 Add Shared Artists & Connect Users
    for artist, users in artist_connections.items():
        G.add_node(
            artist, size=15, color="green", label=f"🎵 {artist}"
        )  # Artists in green

        for user in users:
            G.add_edge(user, artist, weight=1.5)  # Connect users to shared artists

    # 💡 Add Recommended Artists **BUT CONNECT TO THE RECOMMENDING USER**
    recommended_artist_connections = {}  # Store which user recommends which artist
    for user in similar_users:
        user_artists = set(
            user_artist_matrix.loc[user][user_artist_matrix.loc[user] > 0].index
        )
        new_recommendations = (
            user_artists - main_user_artists
        )  # Artists liked by others but not the main user

        for artist in new_recommendations:
            if artist not in recommended_artist_connections:
                recommended_artist_connections[artist] = set()
            recommended_artist_connections[artist].add(
                user
            )  # Connect artist to the recommending user

    for artist, recommending_users in recommended_artist_connections.items():
        G.add_node(
            artist, size=15, color="yellow", label=f"💡 {artist}"
        )  # Recommended artists in yellow

        for user in recommending_users:
            G.add_edge(
                user, artist, weight=1
            )  # CONNECT TO THE RECOMMENDING USER, NOT MAIN USER

    # 🎨 Convert Graph to Plotly
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_size, node_color, text_labels = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]["size"])
        node_color.append(G.nodes[node]["color"])
        text_labels.append(G.nodes[node]["label"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=text_labels,
        marker=dict(
            size=node_size, color=node_color, line=dict(width=2, color="black")
        ),
        textposition="top center",
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        title="🔗 User-Artist Shared Connection Network",
    )

    return fig

def create_user_artist_chord(df, selected_user, top_n_users=10, top_n_artists=10):
    """
    Creates a chord diagram of user-artist relationships.
    """
    # Get top similar users
    similar_users = (
        similarity_df[selected_user]
        .sort_values(ascending=False)
        .head(top_n_users)
        .index
    )

    # Find top artists for each similar user
    user_artist_counts = (
        df[df["user_id"].isin([selected_user] + list(similar_users))]["artistname"]
        .value_counts()
        .head(top_n_artists)
    )

    # Convert to a format for Plotly Sunburst
    data = []
    for user in [selected_user] + list(similar_users):
        for artist in user_artist_counts.index:
            count = len(df[(df["user_id"] == user) & (df["artistname"] == artist)])
            if count > 0:
                data.append({"user": user, "artist": artist, "count": count})

    data_df = pd.DataFrame(data)

    fig = px.sunburst(
        data_df,
        path=["user", "artist"],
        values="count",
        title="🎼 User-Artist Relationship",
    )

    return fig


# ---- USER TAB ----
with tabs[1]:
    if uploaded_file:
        # User Summary
        def user_summary(user_id):
            with st.spinner(f"🔍 Loading user profile for {user_id}..."):
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
            st.success("✅ User profile loaded!")

        # User Input for Recommendation
        st.subheader("🎧 Get Personalized Artist Recommendations")
        selected_user = st.selectbox("Select a User ID", df["user_id"].unique())
        if selected_user:
            user_summary(selected_user)

        # Collaborative Filtering Preparation
        st.subheader("🌐 Building User Similarity Matrix...")
        with st.spinner("📊 Calculating user similarity scores... Please wait!"):
            user_artist_matrix = df.pivot_table(index="user_id", columns="artistname", aggfunc="size", fill_value=0)
            user_artist_sparse = csr_matrix(user_artist_matrix.values)
            similarity_matrix = cosine_similarity(user_artist_sparse)
            similarity_df = pd.DataFrame(similarity_matrix, index=user_artist_matrix.index, columns=user_artist_matrix.index)

            # st.dataframe(similarity_matrix)

            # ---- 📌 Run the User Mapping ----
            with st.expander("📍 2D User Similarity Map"):
                user_embeddings = compute_user_embeddings(similarity_df, method="TSNE")
                plot_user_scatter(user_embeddings)

            # Display in Streamlit
            with st.expander("🙂👨‍🎤🥸 View User-Artist Relationship Network"):
                fig = create_user_artist_shared_network(
                    similarity_df,
                    user_artist_matrix,
                    selected_user,
                    top_n_users=5,
                    top_n_artists=5,
                )
                st.plotly_chart(fig)

            # Display in Streamlit
            with st.expander("🔗 View User-Artist Connection Map"):
                fig = create_user_artist_chord(df, selected_user)
                st.plotly_chart(fig)

            with st.expander("🤝 User-Artist Interaction Grid - Collaborative Filtering Recommendation"):
                if selected_user not in similarity_df.index:
                    st.warning("⚠️ Selected user not found in similarity matrix.")
                else:
                    # 🎛 User Controls
                    col1, col2 = st.columns(2)
                    top_n_users = col1.slider("🔢 Number of Similar Users", 3, 20, 10)
                    top_n_artists = col2.slider("🎼 Number of Artists", 3, 50, 10)

                    # Compute similar users & recommended artists
                    similar_users, recommended_artists = get_similar_users_and_recommendations(
                        selected_user, user_artist_matrix, similarity_df, top_n_users
                    )

                    # Filter matrix for the selected user and similar users
                    filtered_matrix = user_artist_matrix.loc[[selected_user] + list(similar_users)]

                    # User's existing & recommended artists
                    selected_user_artists = set(user_artist_matrix.loc[selected_user][user_artist_matrix.loc[selected_user] > 0].index)

                    # Grouped Artists
                    selected_artists_in_matrix = [a for a in filtered_matrix.columns if a in selected_user_artists]
                    recommended_artists_in_matrix = [a for a in filtered_matrix.columns if a in recommended_artists]

                    # Prioritize order: user's known artists first, followed by new recommendations
                    ordered_artists = selected_artists_in_matrix + [a for a in recommended_artists_in_matrix if a not in selected_artists_in_matrix]
                    ordered_artists = ordered_artists[:top_n_artists]
                    filtered_matrix = filtered_matrix[ordered_artists]

                    # 🎨 Apply New Icons & Labels
                    matrix_data = filtered_matrix.to_numpy()
                    icon_matrix = np.full(matrix_data.shape, "❌", dtype=object)

                    # ✅ User already interacts with artist
                    icon_matrix[matrix_data > 0] = "✔️"

                    # 💡 Highlight AI Recommended Artists
                    for row_idx, user in enumerate(filtered_matrix.index):
                        for col_idx, artist in enumerate(filtered_matrix.columns):
                            if artist in recommended_artists_in_matrix and artist not in selected_user_artists:
                                if matrix_data[row_idx, col_idx] == 0:  # No interaction yet
                                    icon_matrix[row_idx, col_idx] = "💡"

                    # 🖼 Convert to DataFrame
                    grid_df = pd.DataFrame(icon_matrix, index=filtered_matrix.index, columns=filtered_matrix.columns)

                    # 🎨 Style Improvements: Main User Highlight
                    grid_df = grid_df.style.applymap(lambda _: "background-color: #ffeb99; font-weight: bold", subset=pd.IndexSlice[[selected_user], :])

                    # 🖼 Display Updated Grid
                    st.dataframe(grid_df.set_properties(**{"text-align": "center"}))

            with st.expander("🔥 View User Similarity Heatmap"):
                # Select top 10 most similar users
                similar_users = (
                    similarity_df[selected_user].sort_values(ascending=False).head(10).index
                )
                heatmap_data = similarity_df.loc[similar_users, similar_users]

                # 📌 Improve Label Readability
                max_label_length = 8  # Adjust for better readability
                truncated_labels = {
                    user: (
                        user[:max_label_length] + "..."
                        if len(user) > max_label_length
                        else user
                    )
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
                        "🎼 Number of Artists", 3, 50, 10, key="recommendations_artists_slider"
                    )

                    # Use helper function to compute similar users and recommended artists
                    similar_users, recommended_artists = get_similar_users_and_recommendations(
                        selected_user, user_artist_matrix, similarity_df, top_n_users=top_n_users
                    )

                    if not recommended_artists:
                        st.info("⚠️ No recommendations found. The selected user may have already interacted with all artists.")
                    else:
                        st.subheader(f"🎵 Recommended Artists for {selected_user}")

                        # 🎨 UI Enhancement: Display Recommendations in Cards
                        cols = st.columns(min(5, len(recommended_artists)))  # Responsive layout

                        for idx, artist in enumerate(recommended_artists[:top_n_artists]):
                            with cols[idx % len(cols)]:  # Distribute cards evenly
                                st.markdown(
                                    f"""
                                    <div style="
                                        border-radius: 12px;
                                        padding: 15px;
                                        margin: 10px;
                                        background: linear-gradient(135deg, #ff758c, #ff7eb3);
                                        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);
                                        text-align: center;
                                        transition: transform 0.2s ease-in-out;
                                    " onmouseover="this.style.transform='scale(1.05)'" 
                                    onmouseout="this.style.transform='scale(1)'">
                                        <h4 style="color: white; margin: 5px;">🎤 {artist}</h4>
                                        <p style="color: white; margin: 2px;">Suggested by: {', '.join(similar_users)}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        # 📝 Additional Info Section
                        st.info(f"💡 Above are the top {top_n_artists} recommended artists based on similar users.")

        st.success("✅ User similarity matrix ready!")

# ---- ARTISTS TAB ----
with tabs[2]:
    if uploaded_file:
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

        # --- Streamlit App: Fast Artist Clustering Based on Playlist Co-occurrence ---
        st.subheader("🎨 Artist Clustering Based on Playlist Co-occurrence")

        with st.spinner("⏳ Building artist co-occurrence matrix..."):
            artist_cooccurrence = build_artist_cooccurrence_fast(df)
        st.success("✅ Artist co-occurrence matrix built successfully!")

        # Normalize the co-occurrence matrix
        with st.spinner("⏳ Running SVD for dimensionality reduction..."):
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

            with st.expander("Artist vs. Playlists Matrix"):
                # Define sliders to let the user choose how many top playlists and artists to display
                top_n_playlists = st.slider("Select Top N Playlists", min_value=1, max_value=50, value=10, key="playlist_slider")
                top_n_artists = st.slider(
                    "Select Top N Artists",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="artists_slider",
                )

                # Create the artist-playlist matrix (binary presence)
                artist_playlist = df[["artistname", "playlistname"]].drop_duplicates()
                artist_playlist_matrix = pd.crosstab(artist_playlist["artistname"], artist_playlist["playlistname"])

                # Get top playlists and artists based on their frequency in the original DataFrame
                top_playlists = df["playlistname"].value_counts().head(top_n_playlists).index
                top_artists = df["artistname"].value_counts().head(top_n_artists).index

                # Filter the matrix for only these top playlists and artists
                filtered_matrix = artist_playlist_matrix.loc[
                    artist_playlist_matrix.index.isin(top_artists),
                    artist_playlist_matrix.columns.isin(top_playlists)
                ]

                st.dataframe(filtered_matrix)

            with st.expander("🔢 Artist Embedding DataFrame"):
                st.dataframe(artist_reduced_df)
        st.success("✅ Dimensionality reduction completed!")

        # Cluster Artists with K-Means
        with st.spinner("🌀 Running K-Means Clustering..."):
            optimal_k = st.slider("Select Number of Clusters", min_value=2, max_value=20, value=10)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(artist_reduced_df)
            artist_reduced_df["cluster"] = clusters
            # Add a dedicated column for artist names (if not already present)
            artist_reduced_df["artist"] = artist_reduced_df.index

            # Compute and Display Silhouette Score
            sil_score = silhouette_score(
                artist_reduced_df[["Component 1", "Component 2"]], clusters
            )
            st.subheader(f"⭐ Silhouette Score: {sil_score:.2f}")

            # Visualize the Clusters using Plotly
            fig = px.scatter(
                artist_reduced_df,
                x="Component 1",
                y="Component 2",
                color=artist_reduced_df["cluster"].astype(str),
                title="Artist Clusters Based on Playlist Co-occurrence",
                labels={"cluster": "Cluster"},
                hover_name="artist",  # Show artist name on hover
                hover_data={"Component 1": True, "Component 2": True},
            )
            st.plotly_chart(fig)
        st.success(f"✅ Clustering completed! {optimal_k} clusters created.")

        # Allow users to select an artist to see similar artists in the same cluster
        selected_artist = st.selectbox(
            "Select an Artist for Cluster-Based Recommendations", artist_reduced_df.index
        )
        if selected_artist:
            with st.spinner(f"🔍 Finding similar artists to {selected_artist}..."):
                if selected_artist in artist_reduced_df.index:
                    selected_cluster = artist_reduced_df.loc[selected_artist, "cluster"]
                    similar_artists = artist_reduced_df[
                        artist_reduced_df["cluster"] == selected_cluster
                    ].index.tolist()
                    st.write(
                        f"🧑‍🎤👩‍🎤 Artists in the same cluster as {selected_artist}: {similar_artists}"
                    )
                else:
                    st.error(
                        f"Artist {selected_artist} not found in clustering results. Please select another artist."
                    )
