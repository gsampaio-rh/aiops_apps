import streamlit as st
import pandas as pd
import re
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest
import time
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Initialize Knowledge Graph
graph = nx.DiGraph()


# ================================
# 🌟 APP CONFIGURATION
# ================================
st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .stApp { background-color: #f5f5f7; }
        .stDataFrame { border-radius: 10px; }
        .stButton>button { border-radius: 10px; font-size: 16px; }
        .stTextInput>div>div>input { border-radius: 10px; }
        .metric-card { 
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(135deg, #a18cd1, #fbc2eb);
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        .log-classification {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
        }
        .INFO { background-color: #e3f2fd; color: #1565c0; }
        .WARNING { background-color: #fff3e0; color: #e65100; }
        .ERROR { background-color: #ffebee; color: #b71c1c; }
        .CRITICAL { background-color: #fbe9e7; color: #d84315; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🚀 AI-Powered Log Classification & Anomaly Detection")
st.caption("Beautiful, Minimal, Apple-Style UI | Powered by Ollama LLM")

# ================================
# 🔹 LOG PROCESSING FUNCTIONS
# ================================


def parse_log(log_entry):
    """Extracts structured information from a log entry."""
    pattern = (
        r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<hostname>\S+)\s+"
        r"(?P<process>\S+)\[\d+\]:\s+"
        r"(?P<event>.+)"
    )
    match = re.match(pattern, log_entry)
    if match:
        return match.groupdict()
    else:
        return {
            "timestamp": None,
            "hostname": None,
            "process": None,
            "event": log_entry.strip(),
        }


def classify_logs(df_logs):
    """Classifies logs using Ollama API, updating the dataframe dynamically with color coding."""
    endpoint = "http://localhost:11434/api/generate"
    progress_bar = st.progress(0)
    log_table = st.empty()
    log_display = st.empty()

    df_logs["classification"] = "Processing..."
    log_table.dataframe(
        df_logs.style.apply(highlight_rows, axis=1), use_container_width=True
    )

    for i in range(len(df_logs)):
        payload = {
            "model": "mistral",
            "system": 'You are an AI specialized in log classification. Ensure all responses strictly follow this JSON format: {"severity": "INFO/WARNING/ERROR/CRITICAL", "service": "<detected service>", "impact": "<impact assessment>", "region": "<identified region>"}. Classify logs accurately and add contextual metadata. Do not include explanations, only return a valid JSON response.',
            "prompt": df_logs.at[i, "event"],
            "temperature": 0.0,
            "stream": False,
        }
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
        )
        res_json = response.json()
        print(res_json)
        try:
            classification = json.loads(res_json.get("response", "{}"))
            severity = classification.get("severity", "Unknown")  # Store severity in a variable
            df_logs.at[i, "classification"] = severity
            df_logs.at[i, "service"] = classification.get("service", "Unknown")
            df_logs.at[i, "impact"] = classification.get("impact", "Unknown")
            df_logs.at[i, "region"] = classification.get("region", "Unknown")
        except json.JSONDecodeError:
            df_logs.at[i, "classification"] = "Unknown"
            df_logs.at[i, "service"] = "Unknown"
            df_logs.at[i, "impact"] = "Unknown"
            df_logs.at[i, "region"] = "Unknown"

        # Display classification in real-time
        log_display.markdown(
            f"<div class='log-classification {severity}'>{df_logs.at[i, 'event'][:100]}... → <b>{severity}</b></div>",
            unsafe_allow_html=True,
        )

        # Display progress
        progress_bar.progress((i + 1) / len(df_logs))
        log_table.dataframe(
            df_logs.style.apply(highlight_rows, axis=1), use_container_width=True
        )
        time.sleep(0.1)
    return df_logs


def highlight_rows(row):
    """Applies background color to rows based on classification."""
    color_map = {
        "INFO": "background-color: #e3f2fd; color: #1565c0;",
        "WARNING": "background-color: #fff3e0; color: #e65100;",
        "ERROR": "background-color: #ffebee; color: #b71c1c;",
        "CRITICAL": "background-color: #fbe9e7; color: #d84315;",
    }
    return [color_map.get(row.classification, "") for _ in row]


def detect_anomalies(log_texts):
    """Detects anomalies using Isolation Forest."""
    vectorized_logs = np.array([hash(text) % 1000000 for text in log_texts]).reshape(
        -1, 1
    )
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(vectorized_logs)
    return ["Anomaly" if pred == -1 else "Normal" for pred in predictions]


def build_knowledge_graph(df_logs):
    """Builds a knowledge graph connecting logs to services and impacts."""
    for _, row in df_logs.iterrows():
        graph.add_node(
            row["event"],
            service=row["service"],
            severity=row["classification"],
            impact=row["impact"],
            region=row["region"],
        )
        if row["service"] != "Unknown":
            graph.add_edge(row["service"], row["event"], relation="affects")
    return graph


def plot_knowledge_graph(graph):
    """Generates an interactive knowledge graph using Pyvis."""
    net = Network(height="700px", width="100%", notebook=False, directed=True)
    net.from_nx(graph)

    for node in net.nodes:
        node["size"] = 20  # Standardize node sizes
        if "severity" in graph.nodes[node["id"]]:
            severity = graph.nodes[node["id"]]["severity"]
            if severity == "CRITICAL":
                node["color"] = "#ff4c4c"
            elif severity == "ERROR":
                node["color"] = "#ff8c42"
            elif severity == "WARNING":
                node["color"] = "#ffd700"
            else:
                node["color"] = "#4CAF50"
        node["title"] = (
            f"Service: {graph.nodes[node['id']].get('service', 'Unknown')}\nImpact: {graph.nodes[node['id']].get('impact', 'Unknown')}\nRegion: {graph.nodes[node['id']].get('region', 'Unknown')}"
        )

    temp_dir = tempfile.mkdtemp()
    graph_html_path = os.path.join(temp_dir, "graph.html")
    net.save_graph(graph_html_path)
    return graph_html_path


# ================================
# 💂 LOG FILE UPLOAD & PROCESSING
# ================================
uploaded_file = st.file_uploader(
    "🗁 Upload Log File (.log or .txt)", type=["log", "txt"]
)

if uploaded_file:
    log_lines = uploaded_file.getvalue().decode("utf-8").splitlines()
    df_logs = pd.DataFrame({"log_entry": log_lines})
    df_logs = df_logs["log_entry"].apply(parse_log).apply(pd.Series)
    df_logs.dropna(subset=["event"], inplace=True)

    # Display file statistics in metric cards
    st.subheader("📊 Log File Statistics")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='metric-card'>Total Logs<br><b>{len(df_logs)}</b></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='metric-card'>Unique Processes<br><b>{df_logs['process'].nunique()}</b></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='metric-card'>Unique Hostnames<br><b>{df_logs['hostname'].nunique()}</b></div>",
        unsafe_allow_html=True,
    )

    st.subheader("🔹 AI Log Classification")
    # Button to start classification
    if st.button("💡 Start Classification"):

        df_logs = classify_logs(df_logs)

        build_knowledge_graph(df_logs)

        st.subheader("🔹 Knowledge Graph Visualization")
        st.write("Below is the visual representation of the knowledge graph:")

        graph_html_path = plot_knowledge_graph(graph)
        st.components.v1.html(
            open(graph_html_path, "r", encoding="utf-8").read(),
            height=700,
            scrolling=True,
        )

        # Anomaly Detection
        # df_logs["anomaly"] = detect_anomalies(df_logs["event"].tolist())

        # # Display final structured logs
        # st.dataframe(
        #     df_logs.style.apply(highlight_rows, axis=1), use_container_width=True
        # )

else:
    st.info("Please upload a log file to analyze.")
