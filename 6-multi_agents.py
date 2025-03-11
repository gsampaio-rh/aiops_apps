import streamlit as st
import re
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

supervisor_prompt = (
    "You are an AI-powered supervisor responsible for coordinating an automated incident response system. "
    "Your task is to assign troubleshooting steps to specialized agents based on the issue described. "
    "Each agent has a specific role and must be selected sequentially. Return the next step in proper JSON format."
    "\n\n### **Agents Available**:"
    "\n- `log_analyzer`: Fetches server logs and identifies potential errors."
    "\n- `incident_monitor`: Checks system-wide incidents and service status."
    "\n- `fix_suggester`: Suggests possible fixes based on logs and incidents."
    "\n- `action_executor`: Executes the fix and verifies resolution."
    "\n\n### **Instructions**:"
    "\n1. **Analyze the full conversation history**, not just the last user message."
    "\n2. **If a step has already been executed, do not suggest it again unless new issues arise.**"
    "\n3. **If `action_executor` has already restarted a service, verify success before suggesting another action.**"
    "\n4. **If monitoring shows no further errors, conclude the resolution process.**"
    "\n\n### **Loop Prevention Rules**:"
    "\n- If `log_analyzer` has already provided logs, do not call it again unless new errors appear."
    "\n- If `incident_monitor` shows all systems are operational, do not call it again."
    "\n- If `fix_suggester` has already suggested a fix and `action_executor` has executed it, do not suggest the same fix again."
    "\n- If `action_executor` has restarted Nginx and monitoring shows no new issues, return:"
    '\n  `{"thought": "The issue has been fully resolved. No further action is needed.", "next_step": "FINISH"}`'
    "\n- If an issue has been resolved, **do not reanalyze logs or incidents.** Move directly to monitoring."
    "\n\n### **STRICT JSON FORMAT REQUIREMENT**:"
    "\n- Your output **must** be a valid JSON dictionary with two keys: `thought` and `next_step`."
    "\n- **Ensure there are no extra characters, explanations, or formatting issues.**"
    '\n- **Always wrap keys and values in double quotes (`"`), NOT single quotes.**'
    "\n- **DO NOT return any text outside of the JSON block.**"
    "\n\n### **VALID OUTPUT EXAMPLES**:"
    '\n✅ Correct: `{"thought": "The logs indicate an issue with Nginx. Checking incidents next.", "next_step": "incident_monitor"}`'
    '\n❌ Incorrect: `Thought: "The logs indicate an issue with Nginx. Checking incidents next." Next Step: incident_monitor`'
    '\n❌ Incorrect: `{"thought": "Restarting the service might help.", next_step: action_executor}` (missing quotes)'
    '\n❌ Incorrect: `Some text before {"thought": "Checking logs", "next_step": "log_analyzer"} more text after` (extra text before/after JSON)'
    "\n\n### **FINAL CHECK BEFORE RETURNING**:"
    "\n- If your output is NOT a valid JSON dictionary, **correct it before returning.**"
    "\n- If unsure, reformat it using Python’s `json.dumps()` function."
)

def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    user_message = state["messages"][-1].content  # Fix: Access content directly

    # 🔥 Extract the full message history for context
    conversation_history = "\n\n".join([msg.content for msg in state["messages"]])

    display_message(
        f"Input",
        user_message,  # Now includes history
        AGENT_COLORS.get("user_message", AGENT_COLORS["default"]),
    )

    # 🔥 Send the full conversation history to the Supervisor
    response = llm.invoke(
        supervisor_prompt + "\nConversation History:\n" + conversation_history
    )

    # 🔥 Step 1: Clean LLM response
    response_cleaned = response.strip()

    # 🔥 Step 2: Remove **single backticks** if they exist at the start or end
    response_cleaned = re.sub(r"^`|`$", "", response_cleaned)  # Removes ONE leading/trailing backtick

    # 🔥 Step 3: Fix smart quotes (if they exist)
    response_cleaned = response_cleaned.replace("“", '"').replace("”", '"')

    st.code(f"{response_cleaned}")

    try:

        # 🔥 Step 4: Parse JSON safely
        response_json = json.loads(response_cleaned) # Ensure valid JSON

        # 🔥 Validate JSON structure (must contain `thought` and `next_step`)
        if not isinstance(response_json, dict) or "thought" not in response_json or "next_step" not in response_json:
            raise ValueError("Invalid JSON format received.")

        next_step = response_json.get("next_step", "FINISH")

        # 🔥 Append Supervisor thought as a new message (so it's part of history)
        state["messages"].append(
            HumanMessage(content=f"Supervisor Thought: {response_json['thought']}")
        )
        state["messages"].append(HumanMessage(content=f"Next Step: {next_step}"))

        with st.expander("Conversation History"):
            st.markdown(f"{conversation_history}")

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

if st.button("Run AI Agents Graph"):
    with st.spinner("🤖 AI Agents Collaborating..."):
        try:
            step_tracker = st.empty()  # 🔥 Single execution tracker (Fix)
            executed_steps = []  # Store execution order

            for step in graph.stream({"messages": [HumanMessage(content=user_prompt)]}):
                for agent, result in step.items():
                    # Append to execution list
                    executed_steps.append(agent)

                    # 🔥 **Update Execution Step List (Properly)**
                    step_tracker.markdown(
                        "### **Execution Steps**"
                    )  # Render title once
                    step_list = "\n".join(
                        [
                            f"**{idx+1}. {step_name.capitalize()}** ➡️ "
                            for idx, step_name in enumerate(executed_steps)
                        ]
                    )
                    step_tracker.markdown(step_list)  # Update single execution block

                    if result is None:
                        # If there's no content, you can skip or display something else
                        continue

                    display_message(
                        f"{agent.capitalize()} Output",
                        result,
                        AGENT_COLORS.get(agent, AGENT_COLORS["default"]),
                    )

                time.sleep(1)
        except Exception as e:
            display_message("❌ Error Occurred", str(e), "#ffebee")
