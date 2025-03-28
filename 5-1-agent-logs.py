# Streamlit App: AI Agent for Broken Container Fix
import streamlit as st
import subprocess
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_ollama import OllamaLLM

# ------------------- APP CONFIG -------------------
st.set_page_config(page_title="Fix Broken Container Agent", layout="wide")

# ------------------- STYLING -------------------
st.markdown(
    """
    <style>
        .header { text-align: center; margin-top: 30px; font-size: 2em; font-weight: bold; }
        .section-title { font-size: 1.4em; font-weight: 600; margin-top: 30px; }
        .agent-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; font-family: monospace; white-space: pre-wrap; }
        .thought { background-color: #e3f2fd; }
        .action { background-color: #e8f5e9; }
        .observation { background-color: #fff8e1; }
        .final { background-color: #ede7f6; }
        .error { background-color: #ffebee; color: #b71c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
st.markdown(
    "<div class='header'>🧠 AI Agent: Fixing a Broken Container</div>",
    unsafe_allow_html=True,
)

# ------------------- LLM SETUP -------------------
llm = OllamaLLM(model="llama3.1:latest")


# ------------------- TOOL DEFINITIONS -------------------
@tool
def get_container_logs(container_name: str) -> str:
    """Get recent logs from a Podman container."""
    container_name = container_name.strip(
        "\"'"
    )  # Remove extra quotes if passed as a string literal
    result = subprocess.run(
        ["podman", "logs", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logs = result.stdout + result.stderr
    # Return more lines to ensure full visibility
    return logs[-5000:]  # Adjust this if needed


@tool
def suggest_config_fix(logs: str) -> str:
    """Given container logs, suggest a config fix."""
    if "unknown directive" in logs:
        return "The config file contains an unknown directive. Please remove or correct 'invalid_directive'."
    return "No fix identified in the logs."


@tool
def restart_container(container_name: str) -> str:
    """Restart a Podman container."""
    subprocess.run(["podman", "restart", container_name])
    return f"Container {container_name} restarted."


# ------------------- AGENT CONFIG -------------------
tools = [
    Tool(
        name="Get Container Logs",
        func=get_container_logs,
        description="Fetch logs from a specified container.",
    ),
    Tool(
        name="Suggest Config Fix",
        func=suggest_config_fix,
        description="Analyze logs and suggest Nginx config fix.",
    ),
    Tool(
        name="Restart Container",
        func=restart_container,
        description="Restart a Podman container.",
    ),
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

# ------------------- USER INPUT -------------------
st.markdown(
    "<div class='section-title'>💬 Enter Container Name:</div>", unsafe_allow_html=True
)
container_name = st.text_input("Container Name", "broken-nginx")

if st.button("Run AI Agent to Fix Container"):
    with st.spinner("🤖 Agent is diagnosing the issue..."):
        try:
            user_prompt = f"The container '{container_name}' is not responding. Diagnose the issue by analyzing logs and suggest a fix."
            response = agent_executor.invoke(
                {"input": user_prompt}, {"callbacks": [StreamlitCallbackHandler(st)]}
            )
            steps = response.get("intermediate_steps", [])

            st.markdown(
                "<div class='section-title'>🧩 Agent Reasoning Steps:</div>",
                unsafe_allow_html=True,
            )
            for step in steps:
                thought, result = step
                if "Thought:" in str(thought):
                    st.markdown(
                        f"<div class='agent-box thought'><b>Thought:</b> {thought}</div>",
                        unsafe_allow_html=True,
                    )
                if "Action:" in str(thought):
                    st.markdown(
                        f"<div class='agent-box action'><b>Action:</b> {thought}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div class='agent-box observation'><b>Observation:</b> {result}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='section-title'>✅ Final Answer:</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='agent-box final'>{response['output']}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.markdown(
                "<div class='agent-box error'><b>Error:</b> {}</div>".format(str(e)),
                unsafe_allow_html=True,
            )
