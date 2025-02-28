import streamlit as st
import time
import json
import networkx as nx
import matplotlib.pyplot as plt
from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
import requests

# ---- APP CONFIG ----
st.set_page_config(page_title="Autonomous AI Agent Demo", layout="wide")
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
        .header { text-align: center; margin-top: 40px; font-size: 2em; font-weight: bold; }
        .agent-thought { background: #f1f8ff; padding: 10px; border-radius: 5px; font-family: monospace; }
        .agent-action { background: #e3fcef; padding: 10px; border-radius: 5px; font-family: monospace; }
        .agent-observation { background: #fff8e1; padding: 10px; border-radius: 5px; font-family: monospace; }
        .error-message { background: #ffebee; padding: 10px; border-radius: 5px; color: #b71c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- HEADER ----
st.markdown(
    "<h1 class='header'>AI-Powered Incident Management for DevOps</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = Ollama(model="mistral")
memory = ConversationBufferMemory(memory_key="chat_history")


# ---- TOOL FUNCTIONS ----
@tool
def get_server_logs():
    """Fetches recent server logs from a public API."""
    api_url = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs"
    response = requests.get(api_url)
    return (
        response.text[:500]
        if response.status_code == 200
        else "Could not fetch server logs."
    )


@tool
def check_incidents():
    """Fetches current infrastructure incidents from a public API."""
    api_url = "https://www.githubstatus.com/api/v2/status.json"
    response = requests.get(api_url)
    return (
        response.json()
        if response.status_code == 200
        else "Could not fetch incident data."
    )


@tool
def suggest_fix():
    """Suggests a fix based on recent log patterns."""
    return "Based on the logs, the issue appears to be a high error rate in Nginx. Recommended action: Restart the Nginx service."


@tool
def restart_service():
    """Simulated action: Restarting Nginx service."""
    return "Nginx service restarted successfully. Monitoring for further issues."


# ---- INITIALIZE AGENT ----
tools = [
    Tool(
        name="Fetch Server Logs",
        func=get_server_logs,
        description="Retrieves recent logs for troubleshooting.",
    ),
    Tool(
        name="Check Incidents",
        func=check_incidents,
        description="Gets current system status and active incidents.",
    ),
    Tool(
        name="Suggest Fix",
        func=suggest_fix,
        description="Analyzes logs and suggests a resolution.",
    ),
    Tool(
        name="Restart Service",
        func=restart_service,
        description="Executes a restart action for a failing service.",
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
)

# ---- USER INTERACTION ----
user_prompt = st.text_area(
    "📝 Describe your issue (e.g., 'High CPU usage on server-42'):",
    "Nginx service is failing intermittently.",
)

if st.button("Run AI Troubleshooting Agent"):
    with st.spinner("🤖 AI Agent Thinking..."):
        try:
            response = agent.run(user_prompt, callbacks=[StreamlitCallbackHandler(st)])
            st.markdown("### 🤖 Agent Response")
            st.markdown(
                f"<div class='agent-action'><b>Action:</b><br>{response}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown("### ❌ Error Occurred")
            st.markdown(
                f"<div class='error-message'>{str(e)}</div>", unsafe_allow_html=True
            )

    # ---- THOUGHT PROCESS VISUALIZATION ----
    st.markdown("### 🔄 Thought Process")
    for step in agent.memory.chat_memory.messages:
        if "Thought:" in step.content:
            st.markdown(
                f"<div class='agent-thought'><b>Thought:</b><br>{step.content}</div>",
                unsafe_allow_html=True,
            )
        elif "Action:" in step.content:
            st.markdown(
                f"<div class='agent-action'><b>Action:</b><br>{step.content}</div>",
                unsafe_allow_html=True,
            )
        elif "Observation:" in step.content:
            st.markdown(
                f"<div class='agent-observation'><b>Observation:</b><br>{step.content}</div>",
                unsafe_allow_html=True,
            )
        time.sleep(1)

    # ---- GRAPH VISUALIZATION ----
    st.markdown("### 📊 AI Reasoning Flow")
    G = nx.DiGraph()
    G.add_edge("User Request", "Fetch Server Logs")
    G.add_edge("Fetch Server Logs", "Analyze Logs")
    G.add_edge("Analyze Logs", "Suggest Fix")
    G.add_edge("Suggest Fix", "Execute Fix")
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=3000,
        font_size=10,
    )
    st.pyplot(plt)
