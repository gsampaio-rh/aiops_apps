import streamlit as st
import time
import json
import requests
from typing import Literal
import networkx as nx
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain.schema import HumanMessage

# ---- APP CONFIG ----
st.set_page_config(page_title="Multi-Agent AI Supervisor", layout="wide")
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
        .header { text-align: center; margin-top: 40px; font-size: 2em; font-weight: bold; }
        .agent-thought { background: #d6eaf8; padding: 10px; border-radius: 5px; font-family: monospace; }
        .agent-action { background: #e3fcef; padding: 10px; border-radius: 5px; font-family: monospace; }
        .agent-observation { background: #fff8e1; padding: 10px; border-radius: 5px; font-family: monospace; }
        .error-message { background: #ffebee; padding: 10px; border-radius: 5px; color: #b71c1c; }
        .supervisor-input { background: #e3e7fc; padding: 10px; border-radius: 5px; font-family: monospace; }
        .supervisor-output { background: #d6f5d6; padding: 10px; border-radius: 5px; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- HEADER ----
st.markdown(
    "<h1 class='header'>Multi-Agent AI Incident Management with Supervisor</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = OllamaLLM(model="mistral")


# ---- TOOL FUNCTIONS ----
def get_server_logs():
    """Fetches recent server logs."""
    api_url = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs"
    response = requests.get(api_url)
    return (
        response.text[:500] if response.status_code == 200 else "Could not fetch logs."
    )


def check_incidents():
    """Fetches current infrastructure incidents."""
    api_url = "https://www.githubstatus.com/api/v2/status.json"
    response = requests.get(api_url)
    return (
        response.json()
        if response.status_code == 200
        else "Could not fetch incident data."
    )


def suggest_fix():
    """Suggests a fix based on recent log patterns."""
    return "The system is experiencing high 502 errors. Recommended action: Restart the Nginx service."


def restart_service():
    """Simulated action: Restarting Nginx service."""
    return "Nginx service restarted successfully. Monitoring for further issues."


# ---- MULTI-AGENT SYSTEM ----
members = ["log_analyzer", "incident_monitor", "fix_suggester", "action_executor"]
options = members + ["FINISH"]

# ---- SUPERVISOR PROMPT ----
supervisor_prompt = (
    "You are an AI-powered supervisor responsible for coordinating an automated incident response system. "
    "Your task is to assign troubleshooting steps to specialized agents based on the issue described. "
    "Each agent has a specific role and must be selected sequentially. Return the next step in proper JSON format. "
    "\n\nAgents Available:"
    "\n- `log_analyzer`: Fetches server logs and identifies potential errors."
    "\n- `incident_monitor`: Checks system-wide incidents and service status."
    "\n- `fix_suggester`: Suggests possible fixes based on logs and incidents."
    "\n- `action_executor`: Executes the fix and verifies resolution."
    "\n\n### Instructions:"
    "\n1. Analyze the user message."
    "\n2. Determine the most appropriate agent for the next action."
    "\n3. Generate a 'thought' to explain the reasoning behind your selection."
    "\n4. Return the agent selection and thought strictly in the following JSON format:"
    "\n"
    '{"thought": "reasoning", "next_step": "agent_name"}'
    ""
    "\nEnsure that the output is always a valid JSON structure, without extra formatting or characters outside the JSON block."
    '\n5. If all steps are complete, return `{"thought": "The issue has been fully resolved.", "next_step": "FINISH"}`.'
    "\n\n### Example Outputs:"
    '\nUser reports: \'Server is down\' → `{"thought": "To diagnose the issue, fetching logs is the first step.", "next_step": "log_analyzer"}`'
    '\nLogs show high CPU usage → `{"thought": "High CPU usage suggests a system-wide issue. Checking incidents.", "next_step": "incident_monitor"}`'
    '\nIncident detected → `{"thought": "An incident has been identified. A fix should be suggested.", "next_step": "fix_suggester"}`'
    '\nFix suggested → `{"thought": "A fix is available. Executing it now.", "next_step": "action_executor"}`'
    '\nFix executed successfully → `{"thought": "The issue has been fully resolved.", "next_step": "FINISH"}`'
)


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    user_message = state["messages"][-1].content  # Fix: Access content directly
    st.write(user_message)
    response = llm.invoke(supervisor_prompt + "\nUser Message: " + user_message)
    st.code(response)
    try:
        response_json = json.loads(response)
        next_step = response_json.get("next_step", "FINISH")
    except json.JSONDecodeError:
        next_step = "FINISH"
    return Command(goto=next_step if next_step in members else END)


def log_analyzer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    logs = get_server_logs()
    return Command(update={"messages": [HumanMessage(content=logs)]}, goto="supervisor")


def incident_monitor_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    incidents = check_incidents()
    return Command(
        update={"messages": [HumanMessage(content=str(incidents))]}, goto="supervisor"
    )


def fix_suggester_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    fix = suggest_fix()
    return Command(update={"messages": [HumanMessage(content=fix)]}, goto="supervisor")


def action_executor_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = restart_service()
    return Command(
        update={"messages": [HumanMessage(content=result)]}, goto="supervisor"
    )


# ---- BUILDING THE MULTI-AGENT GRAPH ----
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("log_analyzer", log_analyzer_node)
builder.add_node("incident_monitor", incident_monitor_node)
builder.add_node("fix_suggester", fix_suggester_node)
builder.add_node("action_executor", action_executor_node)

graph = builder.compile()


def display_message(label, message, class_name):
    """Utility function to display formatted messages."""
    st.markdown(
        f"<div class='{class_name}'><b>{label}:</b><br>{message}</div>",
        unsafe_allow_html=True,
    )

with st.expander("🥸 Supervisor Prompt"):
    st.code(
        f"""
        {supervisor_prompt} 
        """,
        language="text",
    )

with st.expander("⛓️ Graph"):
    st.code(
        """
        # ---- BUILDING THE MULTI-AGENT GRAPH ----
        builder = StateGraph(MessagesState)
        builder.add_edge(START, "supervisor")
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("log_analyzer", log_analyzer_node)
        builder.add_node("incident_monitor", incident_monitor_node)
        builder.add_node("fix_suggester", fix_suggester_node)
        builder.add_node("action_executor", action_executor_node)

        graph = builder.compile()
        """,
        language="text",
    )

# ---- USER INTERACTION ----
user_prompt = st.text_area(
    "📝 Describe your issue:", "Nginx service is failing intermittently."
)

def display_message(label, message, class_name):
    """Utility function to display formatted messages."""
    if message is not None:
        st.markdown(
            f"<div class='{class_name}'><b>{label}:</b><br>{message}</div>",
            unsafe_allow_html=True,
        )

if st.button("Run AI Supervisor"):
    with st.spinner("🤖 AI Agents Collaborating..."):
        try:
            for step in graph.stream({"messages": [HumanMessage(content=user_prompt)]}):
                for agent, result in step.items():
                    if agent == "supervisor" and result is not None:
                        response_data = json.loads(result)
                        display_message(
                            "Supervisor Thought",
                            response_data.get("thought", "No thought provided"),
                            "agent-thought",
                        )
                        display_message(
                            "Supervisor Decision",
                            response_data.get("next_step", "No decision made"),
                            "supervisor-output",
                        )
                    elif agent != "supervisor":
                        display_message(
                            f"{agent.capitalize()} Output", result, "agent-observation"
                        )
                time.sleep(1)
        except Exception as e:
            display_message("❌ Error Occurred", str(e), "error-message")
