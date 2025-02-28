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
    "<h1 class='header'>O Futuro das Operações – Agentes Autônomos</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = Ollama(model="llama3.1:latest")
memory = ConversationBufferMemory(memory_key="chat_history")


# ---- TOOL FUNCTIONS ----
@tool
def get_weather(city: str):
    """Fetches real-time weather data from a public API."""
    api_url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(api_url)
    return (
        response.text
        if response.status_code == 200
        else "Could not fetch weather data."
    )


@tool
def get_air_quality(city: str):
    """Fetches real-time air quality index (AQI) from a public API."""
    api_url = (
        f"https://api.waqi.info/feed/{city}/?token=demo"  # Replace with a valid API key
    )
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return (
            f"AQI: {data['data']['aqi']}, Status: {data['data']['dominentpol']}"
            if "data" in data
            else "No data available."
        )
    return "Could not fetch AQI data."


# ---- INITIALIZE AGENT ----
tools = [
    Tool(
        name="Get Weather",
        func=get_weather,
        description="Fetches current weather conditions.",
    ),
    Tool(
        name="Get Air Quality",
        func=get_air_quality,
        description="Fetches air quality index (AQI) data.",
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
    "📝 Enter a city to get real-time weather and air quality:", "San Francisco"
)

if st.button("Run Autonomous Agent"):
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
