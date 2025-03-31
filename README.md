# 🚀 Multi-Agent AI Supervisor & Intelligent Systems

## 📌 Overview

This repository is a suite of interactive AI-powered tools designed for:

- **Real-time incident management**
- **Log analysis & classification**
- **Intelligent search with RAG (Retrieval-Augmented Generation)**
- **Music recommendation**
- **Reinforcement learning trainers**
- **Automated codebase clustering**

The core architecture uses **Streamlit** for UI, **LangChain** for intelligent workflows, and a variety of ML/AI models for automation, prediction, and visualization.

---

## 🏷️ Key Features

- 🎧 **Spotify Recommendation Engine** – AI-driven collaborative filtering.
- 🪵 **Log Analysis & Classification** – System anomaly detection via ML & LLMs.
- 🔍 **RAG-based Intelligent Search** – AI-enhanced document querying with vector search.
- 🤖 **Multi-Agent Incident Management** – Autonomous agents for log triage and resolution.
- 🧠 **Reinforcement Learning Trainer** – Q-learning simulation & visualization.
- 📦 **AI Repo Clustering** – Codebase scanning and clustering based on features and language.

---

## 📁 Repository Structure

```text
📦 Project Root
│-- 1-spotify.py                  # Music recommendation system
│-- 2-logs.py                     # Log parsing and anomaly detection
│-- 3-log_classification.py       # Log classification using LLMs
│-- 4-llm_rag.py                  # Intelligent document search with RAG
│-- 5-agents.py                   # Autonomous troubleshooting agents
│-- 5-1-agent-logs.py             # Multi-agent nginx log triage
│-- 6-multi_agents.py             # AI incident resolution with multiple agents
│-- 7-rl.py                       # Reinforcement learning visualization
│-- 8-clustering-repos.py         # Git repo clustering
│-- files/
│   ├── spotify_sample.csv        # Sample music dataset
│   ├── linux_logs.log            # Sample Linux system logs
│   └── README.md                 # Sample README file for RAG demo
│-- requirements.txt              # Python dependencies
│-- README.md                     # This file
```

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/gsampaio-rh/aiops_apps.git
cd aiops_apps
```

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install & Run Ollama (for LLMs)

Several modules use local LLMs powered by **Ollama**.

#### 🧠 Install Ollama:

- Visit: [https://ollama.com](https://ollama.com)
- Download and install for your platform (Mac, Linux, Windows).

#### ▶️ Start Ollama & Pull a Model:

```bash
ollama run llama3
```

Or replace `llama3` with another supported model (e.g., `mistral`, `codellama`).

---

## 🚦 Running Applications

### ▶️ Spotify Recommendation Engine

```bash
streamlit run 1-spotify.py
```

### ▶️ Log Analysis & Anomaly Detection

```bash
streamlit run 2-logs.py
```

### ▶️ Log Classification (LLM-based)

```bash
streamlit run 3-log_classification.py
```

### ▶️ RAG-Based Document Search

```bash
streamlit run 4-llm_rag.py
```

### ▶️ Autonomous AI Agents

```bash
streamlit run 5-agents.py
```

### ▶️ Agent-based Log Triage (w/ Broken Nginx Simulation)

```bash
podman run -d \
  --name broken-nginx \
  -p 8080:80 \
  --volume ./nginx/invalid_nginx.conf:/etc/nginx/nginx.conf:ro \
  nginx

streamlit run 5-1-agent-logs.py
```

### ▶️ Multi-Agent AI Supervisor

```bash
streamlit run 6-multi_agents.py
```

### ▶️ Reinforcement Learning AI Trainer

```bash
streamlit run 7-rl.py
```

### ▶️ Repository Clustering System

```bash
streamlit run 8-clustering-repos.py
```

---

## 📂 Data Samples Included

- `data/spotify_sample.csv` – Example data for music recommender.
- `data/linux_logs.txt` – Sample logs for testing anomaly detection and classification.
- `data/sample_rag_readme.md` – Test document for the RAG intelligent search system.

---

## 🧠 Core Tech Stack

- **Streamlit** – UI framework for fast prototyping.
- **LangChain** – LLM workflow orchestration.
- **scikit-learn, FAISS, pandas, matplotlib** – ML, clustering, and visualization.
- **Ollama** – Local LLM engine.
- **Podman/Nginx** – Simulated production environment for agent testing.

---

## 📝 License

MIT License — free to use, fork, and contribute. Contributions are welcome!
