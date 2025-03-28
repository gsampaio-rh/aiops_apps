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
        .agent-box { padding: 14px 16px; border-radius: 12px; margin-bottom: 16px; font-family: monospace; white-space: pre-wrap; border-left: 5px solid #1976d2; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .thought { background-color: #e3f2fd; border-left-color: #1976d2; }
        .action { background-color: #e8f5e9; border-left-color: #388e3c; }
        .observation { background-color: #fff8e1; border-left-color: #f9a825; }
        .final { background-color: #ede7f6; border-left-color: #7b1fa2; }
        .error { background-color: #ffebee; color: #b71c1c; border-left-color: #c62828; }
        .agent-label { font-weight: bold; font-size: 1rem; margin-bottom: 4px; display: block; }
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


with st.expander("🤖 Agent Overview"):
    st.image(
        "https://github.com/gsampaio-rh/virt-llm-agents/blob/main/images/agent_modules_small.png?raw=true",
        caption="Agent Overview",
    )

with st.expander("🤖 Agent Modules"):
    st.image(
        "https://github.com/gsampaio-rh/virt-llm-agents/blob/main/images/agent.png?raw=true",
        caption="Agent Modules",
    )

with st.expander("🧠 How ReAct Works [Planning Module]"):

    st.write(
        "ReAct (Reasoning + Acting) is a framework where the AI agent iterates between thought, action, and observation."
    )
    st.image(
        "https://github.com/gsampaio-rh/virt-llm-agents/blob/main/images/react-diagram.png?raw=true",
        caption="ReAct Framework",
    )
    st.code(
        """
        ================================ Human Message =================================
        What is 10+10?
        ================================== Ai Message ==================================
        {
            "thought": "The problem requires a basic arithmetic operation, so I will use the 'basic_calculator' tool.",
            "action": "basic_calculator",
            "action_input": {
                "num1": 10,
                "num2": 10,
                "operation": "add"
            }
        }
        ================================ System Message ================================
        The answer is: 20.
        Calculated with basic_calculator.
        ================================== Ai Message ==================================
        {
            "answer": "I have the answer: 20."
        }
        """
    )

with st.expander("🔧 View Tool Implementations"):
    st.code(
        """
@tool
def get_container_logs(container_name: str) -> str:
    container_name = container_name.strip(
        "'"
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
    logs_lower = logs.lower()
    logs_lower = logs_lower.strip(
        "'"
    ) 
    messages = []
    if "fix_invalid_directive" in logs_lower:
        messages.append(
            "Detected 'fix_invalid_directive' error. Please check and remove or correct the invalid directive in nginx.conf."
        )
    if "permission denied" in logs_lower:
        messages.append(
            "Permission issue detected. Ensure the container has proper access rights."
        )
    if "connection refused" in logs_lower:
        messages.append(
            "Connection issue detected. The service may have failed to start or is unreachable."
        )
    if "emerg" in logs_lower:
        messages.append(
            "Critical Nginx startup error detected. Please verify the configuration for syntax errors or unsupported settings."
        )
    return (
        "".join(messages)
        if messages
        else "No recognizable error found. Please check logs manually for further troubleshooting."
    )

@tool
def restart_container(container_name: str) -> str:
    container_name = container_name.strip("'")
    subprocess.run(["podman", "restart", container_name])
    return f"Container {container_name} restarted."

        """,
        language="python",
    )


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
    logs_lower = logs.lower()
    logs_lower = logs_lower.strip("'")
    messages = []
    if "fix_invalid_directive" in logs_lower:
        messages.append(
            "Detected 'fix_invalid_directive' error. Please check and remove or correct the invalid directive in nginx.conf."
        )
    if "permission denied" in logs_lower:
        messages.append(
            "Permission issue detected. Ensure the container has proper access rights."
        )
    if "connection refused" in logs_lower:
        messages.append(
            "Connection issue detected. The service may have failed to start or is unreachable."
        )
    if "emerg" in logs_lower:
        messages.append(
            "Critical Nginx startup error detected. Please verify the configuration for syntax errors or unsupported settings."
        )
    return (
        "\n".join(messages)
        if messages
        else "No recognizable error found. Please check logs manually for further troubleshooting."
    )

@tool
def restart_container(container_name: str) -> str:
    """Restart a Podman container."""
    container_name = container_name.strip("\"'")
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
            user_prompt = f"The container {container_name} is not responding. Diagnose the issue by analyzing logs and suggest a fix."
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
                        f"<div class='agent-box thought'><span class='agent-label'>🧠 Thought</span>{thought}</div>",
                        unsafe_allow_html=True,
                    )
                if "Action:" in str(thought):
                    st.markdown(
                        f"<div class='agent-box action'><span class='agent-label'>⚙️ Action</span>{thought}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div class='agent-box observation'><span class='agent-label'>🔍 Observation</span>{result}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='section-title'>✅ Final Answer:</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='agent-box final'><span class='agent-label'>✅ Result</span>{response['output']}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.markdown(
                f"<div class='agent-box error'><span class='agent-label'>❌ Error</span>{str(e)}</div>",
                unsafe_allow_html=True,
            )

with st.expander("📊 Architecture Overview"):
    st.image(
        "https://raw.githubusercontent.com/gsampaio-rh/virt-llm-agents/4c7358a53b140c75c6c4ad94828b02e7298f0bd4/images/react_flow.png"
    )

with st.expander("🤖 Agent Configuration"):
    st.code(
        """
        from langchain import hub
        from langchain.agents import AgentExecutor, create_react_agent
        
        prompt = hub.pull("hwchase17/react")  # Pulling a standard ReAct prompt
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
        """,
        language="python",
    )

with st.expander("🧩 Prompt + Config Details"):
    st.code(prompt, language="text")
