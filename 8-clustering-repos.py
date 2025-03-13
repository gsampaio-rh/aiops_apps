import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from typing import List

# --------------------------------------------------------------------------------
# 1. Custom Stopwords (additional code-related terms, placeholders, etc.)
# --------------------------------------------------------------------------------
CUSTOM_STOPWORDS = [
    # Code-related or boilerplate terms you want to remove
    "copyright",
    "license",
    "todo",
    "fixme",
    # Common programming placeholders
    "var",
    "func",
    "function",
    "class",
    "def",
    # You can add more domain-specific stopwords here
]


# --------------------------------------------------------------------------------
# Function to find repositories with graphical folder selection
# --------------------------------------------------------------------------------
def find_repositories(root_path):
    repositories = []
    folder_structure = {}

    for root, dirs, files in os.walk(root_path):
        if ".git" in dirs:
            repositories.append(root)
            folder_structure[root] = dirs + files

    return repositories, folder_structure


# --------------------------------------------------------------------------------
# Function to extract summarized file contents
# --------------------------------------------------------------------------------
def extract_summary(file_paths, max_words=100):
    """Reads files and extracts a summarized version for clustering."""
    contents = []
    for path in file_paths:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # Take the first `max_words` words
                    summary = " ".join(content.split()[:max_words])
                    contents.append(summary)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return " ".join(contents)


# --------------------------------------------------------------------------------
# Function to get lines of code (LOC) as a numeric feature (optional improvement)
# --------------------------------------------------------------------------------
def get_loc(file_paths) -> int:
    """Returns a rough count of lines of code for the repository."""
    total_lines = 0
    for path in file_paths:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines += sum(1 for _ in f)  # quick line count
        except:
            pass
    return total_lines


# --------------------------------------------------------------------------------
# Main feature extraction function
# --------------------------------------------------------------------------------
def extract_features(repo_path):
    """Extracts languages, dependencies, code summaries, etc. from a repo."""
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
        ".sh": "ShellScripting",
        ".bat": "BatchFile",
        ".ps1": "PowerShell",
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

    # Detect dependencies (extended to detect Helm, Ansible, etc.)
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

    # Summarize code/content
    features["contents"] = extract_summary(file_paths, max_words=150)

    # Optional numeric feature: lines of code
    features["loc"] = get_loc(file_paths)

    return features


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.set_page_config(page_title="AI-Powered Repository Clustering", layout="wide")
st.title("📊 AI-Powered Repository Clustering")

repo_path = st.text_input("Enter root directory containing repositories:")

if repo_path and os.path.exists(repo_path):
    with st.spinner("🔍 Scanning repositories.."):
        repositories, folder_structure = find_repositories(repo_path)

    if repositories:
        st.write(f"✅ Found {len(repositories)} repositories.")

        # ----------------------------
        # Extracting features
        # ----------------------------
        data = []
        with st.spinner("🔍 Extracting Features..."):
            for repo in repositories:
                repo_name = os.path.basename(repo)
                features = extract_features(repo)
                data.append({"repo": repo_name, "path": repo, **features})

        df = pd.DataFrame(data)

        # ----------------------------
        # Additional Analytics
        # ----------------------------
        with st.spinner("📊 Analyzing Data..."):
            st.subheader("📊 Repository Overview")
            with st.expander("Repository Dataframe"):
                st.dataframe(df.head(100))

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
                # Dependency Distribution
                if "dependencies" in df.columns:
                    dep_counts = (
                        df["dependencies"].str.split(", ").explode().value_counts()
                    )
                    fig_dep = px.bar(
                        x=dep_counts.index,
                        y=dep_counts.values,
                        title="Dependency Usage",
                        labels={"x": "Dependency", "y": "Count"},
                    )
                    st.plotly_chart(fig_dep, use_container_width=True)

        # ------------------------------------------------------------
        # CLUSTERING
        # ------------------------------------------------------------
        with st.spinner("📊 Clustering repositories..."):

            # --------------------------------------------------------
            # 2. Weighted Feature Extraction
            #    We vectorize (languages + dependencies) separately
            #    from code contents, then combine them with weights
            # --------------------------------------------------------
            text_stack_1 = (df["languages"] + ", " + df["dependencies"]).fillna("")
            text_stack_2 = df["contents"].fillna("")

            # TF-IDF vectorizer for languages/dependencies
            vectorizer_stack_1 = TfidfVectorizer(
                stop_words="english",
                max_features=500,
                token_pattern=r"[A-Za-z0-9]+",  # simpler token pattern
                lowercase=True,
            )

            # TF-IDF vectorizer for code contents
            # we add custom stopwords here
            vectorizer_stack_2 = TfidfVectorizer(
                stop_words=("english"),
                max_features=2000,
                token_pattern=r"[A-Za-z0-9]+",
                lowercase=True,
            )

            # Fit transform
            mat_stack_1 = vectorizer_stack_1.fit_transform(text_stack_1)
            mat_stack_2 = vectorizer_stack_2.fit_transform(text_stack_2)

            # Adjust these multipliers as you see fit.
            # Weighted combination of vectors
            # e.g., tech stack might get 1.0 weight, code text 1.0
            weight_stack_1 = 1.0
            weight_stack_2 = 1.0

            # Instead of adding them, horizontally stack them:
            from scipy.sparse import hstack

            combined_features = hstack([
                mat_stack_1.multiply(weight_stack_1),
                mat_stack_2.multiply(weight_stack_2)
            ])

            # Optionally add numeric feature(s) like lines-of-code (loc)
            loc_array = df["loc"].fillna(0).values.reshape(-1, 1)
            loc_array_scaled = loc_array / (loc_array.max() if loc_array.max() else 1)

            final_feature_matrix = hstack([combined_features, loc_array_scaled])

            # Now reduced_features is computed from final_feature_matrix
            svd = TruncatedSVD(n_components=2, random_state=42)
            reduced_features = svd.fit_transform(final_feature_matrix)

            # --------------------------------------------------------
            # 4. Elbow Method and Silhouette Scores (KMeans)
            # --------------------------------------------------------
            with st.expander("📊 Elbow & Silhouette for K-Means"):
                distortions = []
                silhouette_vals = []
                K = range(2, 10)

                for k in K:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_temp = kmeans_temp.fit_predict(reduced_features)
                    # Distortion
                    distortions.append(
                        sum(
                            np.min(
                                cdist(
                                    reduced_features,
                                    kmeans_temp.cluster_centers_,
                                    "euclidean",
                                ),
                                axis=1,
                            )
                        )
                        / reduced_features.shape[0]
                    )
                    # Silhouette
                    silhouette_avg = silhouette_score(reduced_features, cluster_temp)
                    silhouette_vals.append(silhouette_avg)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(K, distortions, "bo-", label="Distortion")
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Distortion")
                ax.set_title("Elbow Method")

                ax2 = ax.twinx()
                ax2.plot(K, silhouette_vals, "ro-", label="Silhouette")
                ax2.set_ylabel("Silhouette Score")

                # Add legends
                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")

                st.pyplot(fig)

            # --------------------------------------------------------
            # 5. Choose the clustering method
            # --------------------------------------------------------
            clustering_method = st.radio(
                "Choose clustering method:",
                ["KMeans", "DBSCAN", "Hierarchical (Agglomerative)"],
            )

            if clustering_method == "KMeans":
                optimal_clusters = st.slider("Select number of clusters (k):", 2, 10, 5)
                model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
                clusters = model.fit_predict(reduced_features)

            elif clustering_method == "DBSCAN":
                eps_val = st.slider("DBSCAN eps:", 0.1, 2.0, 0.5, 0.1)
                min_samples_val = st.slider("DBSCAN min_samples:", 1, 20, 5)
                model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                clusters = model.fit_predict(reduced_features)

            else:  # Hierarchical (Agglomerative)
                n_clusters = st.slider(
                    "Select number of clusters (Agglomerative):", 2, 10, 5
                )
                linkage_method = st.selectbox(
                    "Linkage method:", ["ward", "complete", "average", "single"]
                )
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage=linkage_method
                )
                clusters = model.fit_predict(reduced_features)

            # --------------------------------------------------------
            # 6. Save cluster labels & visualization
            # --------------------------------------------------------
            df["cluster"] = clusters
            df["x"] = reduced_features[:, 0]
            df["y"] = reduced_features[:, 1]

            # Plotly scatter plot
            fig_clusters = px.scatter(
                df,
                x="x",
                y="y",
                color=df["cluster"].astype(str),
                hover_data=["repo", "languages", "dependencies", "loc"],
                title=f"Repository Clusters ({clustering_method})",
                text="repo",
            )
            # Move text a bit
            fig_clusters.update_traces(textposition="top center")
            st.plotly_chart(fig_clusters, use_container_width=True)

    else:
        st.write("❌ No repositories found.")
else:
    st.write("ℹ️ Please enter a valid path.")
