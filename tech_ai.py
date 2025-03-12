import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Function to scan directories for repositories
def find_repositories(root_path):
    repositories = []
    for root, dirs, files in os.walk(root_path):
        if ".git" in dirs:  # Detect repositories
            repositories.append(root)
            dirs.clear()  # Prevent further scanning within this repo
    return repositories


# Function to extract tech stack information
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
        ".yaml": "YAML",
        ".json": "JSON"
    }
    languages = set()

    for file in files:
        _, ext = os.path.splitext(file)
        if ext in extensions:
            languages.add(extensions[ext])

    features["languages"] = ", ".join(languages)

    # Detect dependencies from common files
    dep_files = {
        "package.json": "Node.js",
        "requirements.txt": "Python",
        "Pipfile": "Python",
        "pom.xml": "Java",
        "build.gradle": "Java",
        "Dockerfile": "Docker",
        "Makefile": "Make",
        "composer.json": "PHP",
        "Cargo.toml": "Rust",
    }
    dependencies = set()

    for dep_file in dep_files:
        if dep_file in files:
            dependencies.add(dep_files[dep_file])

    features["dependencies"] = ", ".join(dependencies)

    return features


def cluster_repositories(root_path, n_clusters=10):
    """Clusters repositories based on technology usage."""
    repos = get_repositories(root_path)
    repo_features = {}

    for repo in repos:
        tech_files = list_repo_files(repo)
        features = extract_features(tech_files)
        repo_features[repo] = features

    repo_names = list(repo_features.keys())
    feature_texts = list(repo_features.values())

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(feature_texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clustered_repos = defaultdict(list)
    for repo, label in zip(repo_names, labels):
        clustered_repos[label].append(repo)

    return clustered_repos, labels, repo_names, X


# Streamlit UI
st.title("📊 AI-Powered Repository Clustering")

# User selects directory
repo_path = st.text_input("Or enter root directory containing repositories:")

if repo_path and os.path.exists(repo_path):
    st.write("🔍 Scanning repositories...")
    repos = find_repositories(repo_path)

    if repos:
        st.write(f"✅ Found {len(repos)} repositories.")

        # Extracting features
        data = []
        for repo in repos:
            repo_name = os.path.basename(repo)
            features = extract_features(repo)
            data.append({"repo": repo_name, **features})

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Convert text features to numeric
        vectorizer = TfidfVectorizer()
        feature_matrix = vectorizer.fit_transform(
            df["languages"] + ", " + df["dependencies"]
        )

        # Clustering
        num_clusters = min(5, len(df))  # Prevent issues with small dataset
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(feature_matrix)
        df["cluster"] = clusters

        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(feature_matrix.toarray())
        df["x"] = reduced_features[:, 0]
        df["y"] = reduced_features[:, 1]

        # Visualization
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=df["cluster"].astype(str),
            hover_data=["repo", "languages", "dependencies"],
            title="Repository Clusters",
        )
        st.plotly_chart(fig)
    else:
        st.write("❌ No repositories found.")
else:
    st.write("ℹ️ Please select or enter a valid path.")
