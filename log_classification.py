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
            "system": "You are an AI assistant specialized in log classification. Given a log entry, you must categorize it into one of the following categories: INFO, WARNING, ERROR, CRITICAL. If uncertain, use the closest matching category. Return only the category.",
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
        category = res_json.get("response", "Unknown").strip()
        df_logs.at[i, "classification"] = category

        # Display classification in real-time
        log_display.markdown(
            f"<div class='log-classification {category}'>{df_logs.at[i, 'event'][:100]}... → <b>{category}</b></div>",
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

        # Anomaly Detection
        df_logs["anomaly"] = detect_anomalies(df_logs["event"].tolist())

        # Display final structured logs
        st.dataframe(
            df_logs.style.apply(highlight_rows, axis=1), use_container_width=True
        )

else:
    st.info("Please upload a log file to analyze.")
