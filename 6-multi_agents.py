import streamlit as st
import time
import json
import requests
from typing import Literal
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain.schema import HumanMessage

# Example color choices (pick whichever suits your design needs)
AGENT_COLORS = {
    "supervisor": "#d6eaf8",  # Light Blue
    "log_analyzer": "#d1f2eb",  # Light Teal
    "incident_monitor": "#fcf3cf",  # Light Yellow
    "fix_suggester": "#f9ebea",  # Light Red/Pink
    "action_executor": "#e8daef",  # Light Purple
    "default": "#ff7f00",  # Fallback (Light Orange)
}

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


# ---- INTERACTIVE CHAT UI ----
def display_message(label: str, message: str, bg_color: str) -> None:
    """
    Displays a message in the Streamlit UI with the specified background color.
    """
    st.markdown(
        f"""
        <div style='background: {bg_color}; 
                    padding: 10px; 
                    border-radius: 8px; 
                    margin: 5px 0; 
                    font-family: monospace;'>
            <b>{label}:</b><br>{message}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    display_message(
        f"Input",
        user_message,
        AGENT_COLORS.get("user_message", AGENT_COLORS["default"]),
    )
    response = llm.invoke(supervisor_prompt + "\nUser Message: " + user_message)
    # st.code(response)
    try:
        response_json = json.loads(response)
        next_step = response_json.get("next_step", "FINISH")
        display_message(
            f"Supervisor Output",
            response_json,
            AGENT_COLORS.get("supervisor", AGENT_COLORS["default"]),
        )
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


with st.expander("🥸 Supervisor Prompt"):
    st.code(
        f"""
        {supervisor_prompt} 
        """,
        language="text",
    )


with st.expander("🔧 View Tools Implementations"):
    st.code(
        """
        # ---- TOOL FUNCTIONS ----
        def get_server_logs():
            "Fetches recent server logs."
            api_url = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs"
            response = requests.get(api_url)
            return (
                response.text[:500] if response.status_code == 200 else "Could not fetch logs."
            )


        def check_incidents():
            "Fetches current infrastructure incidents."
            api_url = "https://www.githubstatus.com/api/v2/status.json"
            response = requests.get(api_url)
            return (
                response.json()
                if response.status_code == 200
                else "Could not fetch incident data."
            )


        def suggest_fix():
            "Suggests a fix based on recent log patterns."
            return "The system is experiencing high 502 errors. Recommended action: Restart the Nginx service."


        def restart_service():
            "Simulated action: Restarting Nginx service."
            return "Nginx service restarted successfully. Monitoring for further issues."

        """,
        language="python",
    )

with st.expander("⛓️ Graph"):
    st.code(
        """
        def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
            user_message = state["messages"][-1].content  # Fix: Access content directly
            # st.write(user_message)
            response = llm.invoke(supervisor_prompt + "User Message: " + user_message)
            # st.code(response)
            try:
                response_json = json.loads(response)
                next_step = response_json.get("next_step", "FINISH")
                display_message(
                    f"Supervisor Output",
                    response_json,
                    AGENT_COLORS.get("supervisor", AGENT_COLORS["default"]),
                )
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
        """,
        language="python",
    )

def plot_graph():
    # Define an improved layout for clarity
    pos_improved = {
        "START": (2, 5),
        "supervisor": (2, 3),
        "log_analyzer": (1, 3),
        "incident_monitor": (3, 3),
        "fix_suggester": (1, 2),
        "action_executor": (3, 2),
        "END": (2, 1),
    }

    # Define edges (connections between nodes)
    edges = [
        ("START", "supervisor"),
        ("supervisor", "log_analyzer"),
        ("supervisor", "incident_monitor"),
        ("supervisor", "fix_suggester"),
        ("supervisor", "action_executor"),
        ("log_analyzer", "supervisor"),
        ("incident_monitor", "supervisor"),
        ("fix_suggester", "supervisor"),
        ("action_executor", "supervisor"),
        ("supervisor", "END"),
    ]

    # Create a directed graph
    G_improved = nx.DiGraph()
    G_improved.add_edges_from(edges)

    # Define colors per role
    node_colors = {
        "START": "#1f78b4",  # Blue (Entry point)
        "supervisor": "#33a02c",  # Green (Central Control)
        "log_analyzer": "#a6cee3",  # Light Blue (Analysis)
        "incident_monitor": "#fb9a99",  # Red (Monitoring Issues)
        "fix_suggester": "#fdbf6f",  # Yellow (Fix Recommendations)
        "action_executor": "#b2df8a",  # Light Green (Execution)
        "END": "#ff7f00",  # Orange (Resolution)
    }

    # Assign colors to nodes
    node_color_list = [node_colors[node] for node in G_improved.nodes]

    # Generate figure
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(
        G_improved,
        pos_improved,
        with_labels=True,
        node_color=node_color_list,
        edge_color="gray",
        node_size=2500,
        font_size=10,
        font_weight="bold",
        arrows=True,
        ax=ax,
    )

    # Add labels to edges
    edge_labels = {
        ("START", "supervisor"): "Receive Issue",
        ("supervisor", "log_analyzer"): "Analyze Logs",
        ("supervisor", "incident_monitor"): "Check Incidents",
        ("supervisor", "fix_suggester"): "Suggest Fix",
        ("supervisor", "action_executor"): "Execute Fix",
        ("log_analyzer", "supervisor"): "Send Log Analysis",
        ("incident_monitor", "supervisor"): "Send Incident Report",
        ("fix_suggester", "supervisor"): "Send Fix Suggestion",
        ("action_executor", "supervisor"): "Confirm Execution",
        ("supervisor", "END"): "Issue Resolved",
    }

    nx.draw_networkx_edge_labels(
        G_improved, pos_improved, edge_labels=edge_labels, font_size=6, ax=ax
    )

    # Create legend
    patches = [
        mpatches.Patch(color=color, label=node) for node, color in node_colors.items()
    ]
    plt.legend(handles=patches, loc="lower left", fontsize=9, title="Agent Roles")

    # Title
    plt.title("Enhanced Multi-Agent AI Supervisor Workflow", fontsize=12)

    # Show the improved graph in Streamlit
    st.pyplot(fig)


# Button to display the graph
with st.expander("🗺️ Visualize Graph"):
    plot_graph()

# ---- USER INTERACTION ----
user_prompt = st.text_area(
    "📝 Describe your issue:", "Nginx service is failing intermittently."
)

if st.button("Run AI Supervisor"):
    with st.spinner("🤖 AI Agents Collaborating..."):
        try:
            for step in graph.stream({"messages": [HumanMessage(content=user_prompt)]}):
                for agent, result in step.items():
                    if result is None:
                        # If there's no content, you can skip or display something else
                        continue

                    display_message(
                        f"{agent.capitalize()} Output",
                        result,
                        AGENT_COLORS.get(agent, AGENT_COLORS["default"]),
                    )
                    # Other agents: log_analyzer, incident_monitor, etc.

                time.sleep(1)
        except Exception as e:
            display_message("❌ Error Occurred", str(e), "#ffebee")
