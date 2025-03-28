import streamlit as st
import time
import json
import requests
import matplotlib.pyplot as plt
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_ollama import OllamaLLM

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
    "<h1 class='header'>AI-Powered Agents for Incident Management</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = OllamaLLM(model="mistral")

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
        def get_server_logs():
            "Fetches recent server logs from a public API."
            api_url = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs"
            response = requests.get(api_url)
            return response.text[:500] if response.status_code == 200 else "Could not fetch server logs."
            
        @tool
        def check_incidents():
            "Fetches current infrastructure incidents from a public API."
            api_url = "https://www.githubstatus.com/api/v2/status.json""
            response = requests.get(api_url)
            return (
                response.json()
                if response.status_code == 200
                else "Could not fetch incident data."
            )
        
        @tool
        def suggest_fix():
            "Suggests a fix based on recent log patterns."
            return "Based on the logs, the issue appears to be a high error rate in Nginx. Recommended action: Restart the Nginx service."


        @tool
        def restart_service():
            "Simulated action: Restarting Nginx service."
            return "Nginx service restarted successfully. Monitoring for further issues."
        """,
        language="python",
    )

with st.expander("🤖 ReAct Agent + Tools Workflow"):
    st.image(
        "https://raw.githubusercontent.com/gsampaio-rh/virt-llm-agents/4c7358a53b140c75c6c4ad94828b02e7298f0bd4/images/react_flow.png",
        caption="ReAct Agent + Tools Workflow",
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

with st.expander("📝 View AI Prompt"):
    st.code(
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        You are an expert in composing functions. You are given a question and a set of possible functions.
        Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
        If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
        You should only return the function call in tools call sections.
        
        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        You SHOULD NOT include any other text in the response.
        
        Here is a list of functions in JSON format that you can invoke.
        
        [
            {
                "name": "get_server_logs",
                "description": "Fetches recent server logs from a public API.",
                "parameters": {
                    "type": "dict",
                    "required": [],
                    "properties": {}
                }
            },
            {
                "name": "check_incidents",
                "description": "Gets current system status and active incidents.",
                "parameters": {
                    "type": "dict",
                    "required": [],
                    "properties": {}
                }
            },
            {
                "name": "suggest_fix",
                "description": "Analyzes logs and suggests a resolution.",
                "parameters": {
                    "type": "dict",
                    "required": [],
                    "properties": {}
                }
            },
            {
                "name": "restart_service",
                "description": "Executes a restart action for a failing service.",
                "parameters": {
                    "type": "dict",
                    "required": [],
                    "properties": {}
                }
            }
        ]<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        The server is experiencing high error rates.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [get_server_logs(), check_incidents()]
        """,
        language="text",
    )


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

# ---- CREATE REACT AGENT ----
prompt = hub.pull("hwchase17/react")  # Pulling a standard ReAct prompt
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# ---- Header ----
st.markdown(
    "<h1 class='header'>Prompt the Agent with something...</h1>", unsafe_allow_html=True
)
# ---- USER INTERACTION ----
user_prompt = st.text_area(
    "📝 Describe your issue (e.g., 'High CPU usage on server-42'):",
    "Nginx service is failing intermittently.",
)

if st.button("Run AI Troubleshooting Agent"):
    with st.spinner("🤖 AI Agent Thinking..."):
        try:
            response = agent_executor.invoke(
                {"input": prompt}, {"callbacks": [StreamlitCallbackHandler(st)]}
            )
            st.markdown("### 🤖 Agent Response")
            st.markdown(
                f"<div class='agent-action'><b>Action:</b><br>{response['output']}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown("### ❌ Error Occurred")
            st.markdown(
                f"<div class='error-message'>{str(e)}</div>", unsafe_allow_html=True
            )
