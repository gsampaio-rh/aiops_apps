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

# ---- APP CONFIG ----
st.set_page_config(page_title="Autonomous AI Agent Demo", layout="wide")
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
        .header { text-align: center; margin-top: 40px; font-size: 2em; font-weight: bold; }
        .agent-thought { background: #f1f8ff; padding: 10px; border-radius: 5px; }
        .agent-action { background: #e3fcef; padding: 10px; border-radius: 5px; }
        .error-message { background: #ffebee; padding: 10px; border-radius: 5px; color: #b71c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- HEADER ----
st.markdown(
    "<h1 class='header'>Agents</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = Ollama(model="llama3.1:latest")
memory = ConversationBufferMemory(memory_key="chat_history")


# ---- TOOL FUNCTIONS ----
@tool
def check_server_status(input_text: str = ""):
    """Simulated function that returns the server status."""
    return "Server CPU at 95%. Possible overload detected."


@tool
def restart_server(input_text: str = ""):
    """Simulated function that restarts a failing server."""
    return "Server restarted successfully. Monitoring for further issues."


# ---- INITIALIZE AGENT ----
tools = [
    Tool(
        name="check_server_status",
        func=check_server_status,
        description="Checks the server health.",
    ),
    Tool(
        name="restart_server",
        func=restart_server,
        description="Restarts the server if needed.",
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
    "📝 Enter an operational issue (e.g., 'Fix high CPU usage'):",
    "Check the server health and resolve any issues.",
)

if st.button("Run Autonomous Agent"):
    with st.spinner("🤖 AI Agent Thinking..."):
        try:
            response = agent.run(user_prompt, callbacks=[StreamlitCallbackHandler(st)])
            st.markdown("### 🤖 Agent Response")
            st.markdown(
                f"<div class='agent-action'>{response}</div>", unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown("### ❌ Error Occurred")
            st.markdown(
                f"<div class='error-message'>{str(e)}</div>", unsafe_allow_html=True
            )

    # ---- THOUGHT PROCESS VISUALIZATION ----
    st.markdown("### 🔄 Thought Process")
    for step in agent.memory.chat_memory.messages:
        st.markdown(
            f"<div class='agent-thought'>🧠 {step.content}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)

