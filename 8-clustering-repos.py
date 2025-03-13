import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# Function to find repositories with graphical folder selection
def find_repositories(root_path):
    repositories = []
    folder_structure = {}

    for root, dirs, files in os.walk(root_path):
        if ".git" in dirs:
            repositories.append(root)
            folder_structure[root] = dirs + files

    return repositories, folder_structure


def extract_features(repo_path):
    features = {}
    files = os.listdir(repo_path)

    # Detect programming languages based on file extensions
    extensions = {
        ".py": "Python",
        ".js": "JavaScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".rb": "Ruby",
        ".php": "PHP",
        ".ts": "TypeScript",
        ".go": "Go",
        ".rs": "Rust",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".yml": "YAML",
        ".tf": "Terraform",
        ".yaml": "YAML",
        ".json": "JSON",
        ".md": "Markdown",
        ".txt": "Text",
        ".csv": "CSV",
        ".html": "HTML",
        ".css": "CSS",
        ".sh": "Shell Scripting",
        ".bat": "Batch File",
        ".ps1": "PowerShell Scripting",
    }
    languages = set()
    file_paths = []

    for file in files:
        file_path = os.path.join(repo_path, file)
        # Skip directories to avoid IsADirectoryError
        if os.path.isfile(file_path):
            file_paths.append(file_path)
            _, ext = os.path.splitext(file)
            if ext in extensions:
                languages.add(extensions[ext])

    features["languages"] = ", ".join(languages)

    # Detect dependencies (extended to detect Helm)
    dep_files = {
        "package.json": "Node.js",
        "requirements.txt": "Python",
        "Pipfile": "Python",
        "pom.xml": "Java",
        "Dockerfile": "Docker",
        "Makefile": "Make",
        "composer.json": "PHP",
        "Cargo.toml": "Rust",
        "Chart.yaml": "Helm",
        "values.yaml": "Helm",
        "Jenkinsfile": "Jenkins",
        "main.tf": "Terraform",
        "playbook.yaml": "Ansible",
    }
    dependencies = set()

    for dep_file, dep_label in dep_files.items():
        if dep_file in files:
            dependencies.add(dep_label)

    features["dependencies"] = ", ".join(dependencies)

    # Extract summarized file contents (only for actual files)
    features["contents"] = extract_summary(file_paths)

    return features


# Function to extract summarized file contents
def extract_summary(file_paths):
    """Reads files and extracts a summarized version for clustering."""
    contents = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                summary = " ".join(
                    content.split()[:100]
                )  # Only keep the first 100 words
                contents.append(summary)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return " ".join(contents)


# Streamlit UI
st.set_page_config(page_title="AI-Powered Repository Clustering", layout="wide")
st.title("📊 AI-Powered Repository Clustering")

repo_path = st.text_input("Enter root directory containing repositories:")

if repo_path and os.path.exists(repo_path):
    with st.spinner("🔍 Scanning repositories.."):
        repositories, folder_structure = find_repositories(repo_path)

    if repositories:
        st.write(f"✅ Found {len(repositories)} repositories.")

        # Extracting features
        data = []
        with st.spinner("🔍 Extract Features..."):
            for repo in repositories:
                repo_name = os.path.basename(repo)
                features = extract_features(repo)
                data.append({"repo": repo_name, **features})

        df = pd.DataFrame(data)

        with st.spinner("📊 Analysing data..."):
            # Additional Analytics
            st.subheader("📊 Repository Overview")
            with st.expander("Repository Dataframe"):
                st.dataframe(df.head(100))

            # Layout columns for dataset stats
            col1, col2 = st.columns(2)

            with col1:
                # Language Distribution
                lang_counts = df["languages"].str.split(", ").explode().value_counts()
                fig_lang = px.pie(
                    values=lang_counts.values,
                    names=lang_counts.index,
                    title="Language Distribution",
                )
                st.plotly_chart(fig_lang, use_container_width=True)

            with col2:
                # Dependency Distribution (if available)
                if "dependencies" in df.columns:
                    dep_counts = df["dependencies"].str.split(", ").explode().value_counts()
                    fig_dep = px.bar(
                        x=dep_counts.index,
                        y=dep_counts.values,
                        title="Dependency Usage",
                        labels={"x": "Dependency", "y": "Count"},
                    )
                    st.plotly_chart(fig_dep, use_container_width=True)

        with st.spinner("📊 Clustering repositories..."):
            # Convert text features to numeric
            vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)  # Limit features to reduce memory usage
            feature_matrix = vectorizer.fit_transform(df['languages'] + ', ' + df['dependencies'] + ', ' + df['contents'])

            # Reduce dimensionality using TruncatedSVD (LSA)
            svd = TruncatedSVD(n_components=2, random_state=42)
            reduced_features = svd.fit_transform(feature_matrix)

            # Determine the optimal number of clusters using the Elbow Method
            distortions = []
            K = range(2, 10)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(reduced_features)
                distortions.append(sum(np.min(cdist(reduced_features, kmeans.cluster_centers_, 'euclidean'), axis=1)) / reduced_features.shape[0])

            with st.expander("📊 Elbow Method for Optimal k"):
                # Plot elbow method
                fig, ax = plt.subplots()
                ax.plot(K, distortions, 'bo-')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Distortion')
                ax.set_title('Elbow Method for Optimal k')
                st.pyplot(fig)

            # Choose the best clustering method
            clustering_method = st.radio(
                "Choose clustering method:", ["KMeans", "DBSCAN"]
            )

            if clustering_method == "KMeans":
                optimal_clusters = st.slider("Select number of clusters:", 2, 10, 5)
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(reduced_features)
            else:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(reduced_features)

            df['cluster'] = clusters
            df['x'] = reduced_features[:, 0]
            df['y'] = reduced_features[:, 1]

            # Scatter plot with repo names as labels
            fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str),
                            hover_data=['repo', 'languages'], title="Repository Clusters",
                            text=df['repo'])
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

            # Visualization
            # fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str), hover_data=['repo', 'languages', 'dependencies'], title="Repository Clusters")
            # st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("❌ No repositories found.")
else:
    st.write("ℹ️ Please enter a valid path.")
