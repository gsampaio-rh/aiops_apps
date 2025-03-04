# 🚀 Multi-Agent AI Supervisor & Intelligent Systems

## 📌 Overview

This repository contains multiple AI-driven interactive applications designed for **real-time incident management, log analysis, recommendation systems, and intelligent search**. The core applications demonstrate **multi-agent collaboration, RAG (Retrieval-Augmented Generation), and advanced machine learning** for predictive analysis.

Each module is built with **Streamlit for UI**, **Langchain for AI workflows**, and various ML/AI techniques for automation, visualization, and intelligence.

## 🏗️ Features

- **Multi-Agent Incident Management**: AI agents coordinate log analysis, incident detection, and resolution.
- **RAG-based Intelligent Search**: AI retrieves, ranks, and presents information from documents.
- **Spotify Recommendation Engine**: AI-based music recommendations using collaborative filtering.
- **Log Analysis & Anomaly Detection**: ML-based system for detecting unusual patterns in system logs.
- **Graph-Based Event Transitions**: Visualizations of event relationships in log data.
- **Predictive AI for System Events**: Uses ML models to forecast future log-based incidents.

---

## 📂 Repository Structure

```text
📦 Project Root
│-- multi-agents.py          # AI-driven incident management with multi-agents
│-- agents.py                # Standalone AI agents for task automation
│-- llm_rag.py               # Retrieval-Augmented Generation (RAG) implementation
│-- logs.py                  # Log processing and event prediction
│-- spotify.py               # Collaborative filtering-based music recommendations
│-- requirements.txt          # Dependencies for running the project
│-- README.md                # Project documentation
```

---

## 🚀 Setup & Installation

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### **2️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **3️⃣ Run Each Application**

#### **Multi-Agent AI Supervisor**

```sh
streamlit run multi-agents.py
```

#### **RAG-Based Search**

```sh
streamlit run llm_rag.py
```

#### **Log Analysis & Anomaly Detection**

```sh
streamlit run logs.py
```

#### **Spotify Recommendation System**

```sh
streamlit run spotify.py
```

---

## 🛠️ Core Functionalities

### **🔹 Multi-Agent AI Supervisor (multi-agents.py)**

A real-time **incident management system** where AI agents analyze logs, detect incidents, suggest fixes, and execute resolutions. Built with **Langchain, Streamlit, and a multi-agent graph framework.**

🔹 **Agents:**

- `log_analyzer`: Fetches logs from external sources.
- `incident_monitor`: Checks for service disruptions.
- `fix_suggester`: Recommends solutions based on logs.
- `action_executor`: Applies fixes to resolve issues.

---

### **🔹 RAG-Based Intelligent Search (llm_rag.py)**

An **AI-powered retrieval system** that enhances LLM responses with external document search. Uses **FAISS for vector storage** and **Langchain document loaders.**

🔹 **Key Features:**

- Upload and process Markdown files for search.
- Uses embeddings to rank document relevance.
- Visualizes attention matrices and search queries.

---

### **🔹 Log Analysis & Anomaly Detection (logs.py)**

Processes logs, extracts patterns, and visualizes system anomalies using ML.

🔹 **Capabilities:**

- Parses system logs into structured formats.
- Identifies patterns using **TF-IDF & Clustering.**
- Predicts future incidents based on past trends.

---

### **🔹 Spotify Recommendation Engine (spotify.py)**

A **collaborative filtering** AI system that recommends artists based on user preferences and listening history.

🔹 **Methods Used:**

- **User-based Filtering:** Finds similar users for recommendations.
- **Item-based Filtering:** Identifies music trends from similar artists.
- **Clustering & Co-occurrence Analysis:** Detects playlist-based patterns.

## 📝 License

This project is **MIT Licensed**—open for contributions and modifications.
