# 🚀 Multi-Agent AI Supervisor & Intelligent Systems

## 📌 Overview

This repository contains multiple AI-driven interactive applications designed for real-time incident management, log analysis, recommendation systems, intelligent search, and reinforcement learning. The core applications demonstrate multi-agent collaboration, RAG (Retrieval-Augmented Generation), and advanced machine learning for predictive analysis.

Each module is built with Streamlit for UI, Langchain for AI workflows, and various ML/AI techniques for automation, visualization, and intelligence.

## 🏷️ Features

- **Spotify Recommendation Engine**: AI-based music recommendations using collaborative filtering.
- **Log Analysis & Classification**: ML-based system for detecting unusual patterns in system logs.
- **RAG-based Intelligent Search**: AI retrieves, ranks, and presents information from documents.
- **Multi-Agent Incident Management**: AI agents coordinate log analysis, incident detection, and resolution.
- **Reinforcement Learning AI Trainer**: AI-driven reinforcement learning system for training and optimizing strategies.
- **AI Repository Clustering**: Automated scanning of local Git repositories to extract features, visualize distributions, and cluster repos by similarity.

---

## 💂️️ Repository Structure

```text
📦 Project Root
│-- 1-spotify.py              # Collaborative filtering-based music recommendations
│-- 2-logs.py                 # Log processing and event detection
│-- 3-log_classification.py   # Log classification using LLMs
│-- 4-llm_rag.py              # Retrieval-Augmented Generation (RAG) implementation
│-- 5-agents.py               # Standalone AI agents for task automation
│-- 6-multi_agents.py         # AI-driven incident management with multi-agents
│-- 7-rl.py                   # Reinforcement Learning-based AI Trainer
│-- 8-clustering-repos.py     # AI-powered repository clustering and feature extraction
│-- requirements.txt          # Dependencies for running the project
│-- README.md                # Project documentation
│-- data/                     # Data storage directory
```

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Run Each Application

#### Spotify Recommendation System

```sh
streamlit run 1-spotify.py
```

#### Log Analysis & Anomaly Detection

```sh
streamlit run 2-logs.py
```

#### Log Classification

```sh
streamlit run 3-log_classification.py
```

#### RAG-Based Search

```sh
streamlit run 4-llm_rag.py
```

#### Autonomous AI Agents

```sh
streamlit run 5-agents.py
```

#### Multi-Agent AI Supervisor

```sh
streamlit run 6-multi_agents.py
```

#### Reinforcement Learning AI Trainer

```sh
streamlit run 7-rl.py
```

#### AI-Powered Repository Clustering

```sh
streamlit run 8-clustering-repos.py
```

---

## 🛠️ Core Functionalities

### 🔹 Spotify Recommendation Engine (1-spotify.py)

A collaborative filtering AI system that recommends artists based on user preferences and listening history.

🔹 Methods Used:

- User-based Filtering: Finds similar users for recommendations.
- Item-based Filtering: Identifies music trends from similar artists.
- Clustering & Co-occurrence Analysis: Detects playlist-based patterns.

### 🔹 Log Analysis & Anomaly Detection (2-logs.py)

Processes logs, extracts patterns, and visualizes system anomalies using ML.

🔹 Capabilities:

- Parses system logs into structured formats.
- Identifies patterns using TF-IDF & Clustering.
- Predicts future incidents based on past trends.

### 🔹 Log Classification (3-log_classification.py)

Using LLMs to transform unstructured data into structured data.

🔹 Features:

- Uses LLM to classify data.
- Applies supervised learning models.
- Enhances automated troubleshooting workflows.

### 🔹 RAG-Based Intelligent Search (4-llm_rag.py)

An AI-powered retrieval system that enhances LLM responses with external document search. Uses FAISS for vector storage and Langchain document loaders.

🔹 Key Features:

- Upload and process Markdown files for search.
- Uses embeddings to rank document relevance.
- Visualizes attention matrices and search queries.

### 🔹 Autonomous AI Agents (5-agents.py)

Standalone AI agents that handle automated troubleshooting, log analysis, and system monitoring using AI-powered decision-making.

### 🔹 Multi-Agent AI Supervisor (6-multi_agents.py)

A real-time incident management system where AI agents analyze logs, detect incidents, suggest fixes, and execute resolutions. Built with Langchain, Streamlit, and a multi-agent graph framework.

🔹 Agents:

- `log_analyzer`: Fetches logs from external sources.
- `incident_monitor`: Checks for service disruptions.
- `fix_suggester`: Recommends solutions based on logs.
- `action_executor`: Applies fixes to resolve issues.

### 🔹 Reinforcement Learning AI Trainer (7-rl.py)

A reinforcement learning-based AI system that optimizes algorithmic strategies for solving computational problems.

🔹 Features:

- Uses Q-learning to train AI models for problem-solving.
- Evaluates multiple AI-generated solutions in real-time.
- Implements reinforcement rewards to enhance decision-making.
- Supports real-time visualization of training progress.

### 🔹 AI-Powered Repository Clustering (8-clustering-repos.py)

Scans a directory containing multiple Git repositories, extracts code features (languages, dependencies, lines of code, etc.), and uses machine learning to cluster similar repositories.

**Highlights**:

- Automatically detects `.git` folders.
- Identifies programming languages based on file types.
- Extracts dependencies from files (e.g., `requirements.txt`, `package.json`).
- Summarizes code snippets for text-based clustering.
- Provides advanced clustering methods (K-Means, DBSCAN, Hierarchical).
- Visualizes clusters in 2D using dimensionality reduction (SVD).

---

## 📝 License

This project is MIT Licensed—open for contributions and modifications.
