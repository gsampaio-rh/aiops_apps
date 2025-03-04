import streamlit as st
import pandas as pd
import re
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest

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
        .SECURITY { background-color: #ede7f6; color: #4527a0; }
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


def classify_logs(logs):
    """Classifies logs using Ollama API."""
    endpoint = "http://localhost:11434/api/generate"
    classifications = []
    for log in logs:
        payload = {
            "model": "mistral",
            "system": "You are an AI assistant classifying log messages. Assign one of the following categories: INFO, WARNING, ERROR, CRITICAL, SECURITY. Return only the category.",
            "prompt": log,
            "temperature": 0.0,
            "stream": False,
        }
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
        )
        res_json = response.json()
        category = res_json.get("response", "Unknown").strip()
        classifications.append((log, category))
    return classifications


def detect_anomalies(log_texts):
    """Detects anomalies using Isolation Forest."""
    vectorized_logs = np.array([hash(text) % 1000000 for text in log_texts]).reshape(
        -1, 1
    )
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(vectorized_logs)
    return ["Anomaly" if pred == -1 else "Normal" for pred in predictions]


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

    st.subheader("🔹 Extracted Logs ")
    # Display sample logs
    with st.expander("📄 View Sample Logs"):
        st.dataframe(df_logs.head(), use_container_width=True)

    # AI Classification
    st.subheader("🔹 AI Log Classification")
    classified_logs = classify_logs(df_logs["event"].tolist())

    for log, category in classified_logs:
        st.markdown(
            f"<div class='log-classification {category}'>{log[:100]}... → <b>{category}</b></div>",
            unsafe_allow_html=True,
        )

    # Anomaly Detection
    df_logs["anomaly"] = detect_anomalies(df_logs["event"].tolist())

    # Display structured logs
    st.dataframe(df_logs, use_container_width=True)

    # ================================
    # 📊 DATA VISUALIZATION
    # ================================
    st.subheader("📊 Log Classification Distribution")
    class_counts = df_logs["classification"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Classification")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("🚨 Anomaly Detection")
    anomaly_counts = df_logs["anomaly"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values, palette="Reds", ax=ax)
    ax.set_xlabel("Anomaly Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.info("Please upload a log file to analyze.")
