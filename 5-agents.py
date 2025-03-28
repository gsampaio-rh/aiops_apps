# Refactored Streamlit App: Agentic Workflow Demo
import streamlit as st
import requests
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_ollama import OllamaLLM

# ------------------- APP CONFIG -------------------
st.set_page_config(page_title="AI Agent Demo", layout="wide")

# ------------------- STYLING -------------------
st.markdown(
    """
    <style>
        .header { text-align: center; margin-top: 30px; font-size: 2em; font-weight: bold; }
        .section-title { font-size: 1.4em; font-weight: 600; margin-top: 30px; }
        .agent-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; font-family: monospace; }
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
    "<div class='header'>🧠 AI Agent: Thinking Through Incident Response</div>",
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

# ------------------- TOOL DEFINITIONS -------------------
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


# ------------------- AGENT CONFIG -------------------
tools = [
    Tool(
        name="Fetch Server Logs",
        func=get_server_logs,
        description="Retrieves logs for troubleshooting.",
    ),
    Tool(
        name="Check Incidents",
        func=check_incidents,
        description="Gets current system status.",
    ),
    Tool(
        name="Suggest Fix",
        func=suggest_fix,
        description="Suggests resolutions for known issues.",
    ),
    Tool(
        name="Restart Service",
        func=restart_service,
        description="Simulates restarting a service.",
    ),
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

# ------------------- USER INPUT -------------------
st.markdown(
    "<div class='section-title'>💬 Describe the issue:</div>", unsafe_allow_html=True
)
user_input = st.text_area(
    "Example: 'Nginx service is failing intermittently.'",
    "Nginx service is failing intermittently.",
)

if st.button("Run AI Agent"):
    with st.spinner("🤖 Thinking..."):
        try:
            response = agent_executor.invoke(
                {"input": user_input}, {"callbacks": [StreamlitCallbackHandler(st)]}
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

# ------------------- OPTIONAL: DIAGRAM + TABS -------------------
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
