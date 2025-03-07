import streamlit as st
import time
import json
import numpy as np
import random
import tracemalloc
import re
import ast
import matplotlib.pyplot as plt
import requests
from langchain.llms import Ollama

if "q_table" not in st.session_state:
    st.session_state.q_table = {}  # { state (test case): {action: Q-value} }


def update_q_table(test_case, action, reward, alpha=0.1, gamma=0.9):
    """Updates the Q-table using the Q-learning formula."""
    if test_case not in st.session_state.q_table:
        st.session_state.q_table[test_case] = {}

    # Get current Q-value for (test_case, action), default to 0 if new
    current_q = st.session_state.q_table[test_case].get(action, 0)

    # Estimate future rewards (best known reward for this test case)
    max_future_q = max(st.session_state.q_table[test_case].values(), default=0)

    # Apply the Q-learning update rule
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)

    # Store updated Q-value
    st.session_state.q_table[test_case][action] = new_q


# ---- APP CONFIG ----
st.set_page_config(page_title="AI Learning: Reinforcement Training", layout="wide")

# ---- STYLING ----
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
        .header { text-align: center; font-size: 2em; font-weight: bold; }
        .highlight { background: #e3f2fd; padding: 10px; border-radius: 5px; }
        .correct { color: green; font-weight: bold; }
        .incorrect { color: red; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- HEADER ----
st.markdown(
    "<h1 class='header'>🤖 AI Learning: Reinforcement Training</h1>",
    unsafe_allow_html=True,
)

# ---- LLM SETUP ----
llm = Ollama(model="mistral")

# ---- TEST CASES ----
TEST_CASES = [
    ([1, 2, 3, 4, 5], 4),
    ([10, 10, 9, 8], 9),
    ([5, 5, 5, 5], -1),
    ([100, 90, 80, 70, 60], 90),
    ([3], -1),
]

# ---- AI STRATEGIES ----
actions = [
    "Sort and find second last",
    "Single pass scan with two variables",
    "Use set to remove duplicates, then find max",
    "Brute force nested loop approach",
    "Alternative logic-based approach",
]


# ---- AI CODE GENERATION ----
def generate_solutions():
    """Streams AI solutions for selected strategies."""
    solutions = {}

    for action in selected_actions:  # Use only user-selected strategies
        with st.expander(f"✨ Strategy: {action}"):
            with st.spinner(f"🔄 Generating solution for **{action}**..."):
                placeholder = st.empty()

                prompt = f"""
                Write a Python function named `second_largest` that finds the second largest distinct element in an array.
                Use this approach: {action}. Ensure the function handles edge cases properly.
                Only return the function implementation, no explanations or test cases.
                """

                # Stream response
                response = ""
                for chunk in llm.stream(prompt):  # Streaming the response
                    response += chunk
                    placeholder.markdown(
                        f"```python\n{response}\n```"
                    )  # Update UI in real-time

                # Extract and finalize the function code
                match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                solutions[action] = (
                    match.group(1).strip() if match else response.strip()
                )

                # Replace placeholder with final code block
                placeholder.code(solutions[action], language="python")

    return solutions

def evaluate_solution(solution_code, array, expected_output, action):
    """Runs and evaluates the AI-generated solution and updates Q-table."""
    if not solution_code:
        update_q_table(tuple(array), action, -10)  # Penalize missing solutions
        return -10

    try:
        local_scope = {}
        exec(solution_code, globals(), local_scope)
        second_largest = local_scope.get("second_largest")

        if not second_largest:
            raise ValueError("Function 'second_largest' not found in generated code")

        tracemalloc.start()
        start_time = time.time()
        result = second_largest(array)
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compute reward (same advanced strategy as before)
        accuracy = 1.0 if result == expected_output else 0
        time_efficiency = max(0, 1 - (execution_time / 0.001))
        memory_efficiency = max(0, 1 - (peak / 1024 / 10))
        complexity_penalty = min(1, len(array) / 100)
        edge_case_penalty = (
            -2 if len(array) < 2 else (-1 if len(set(array)) == 1 else 0)
        )

        reward = (
            accuracy * 10
            + time_efficiency * 50
            + memory_efficiency * 30
            - complexity_penalty * 10
            + edge_case_penalty
        )

        # Update Q-table with the new reward
        update_q_table(tuple(array), action, reward)

        return reward

    except Exception as e:
        update_q_table(tuple(array), action, -10)  # Penalize errors
        return -10


def select_best_strategy(test_case):
    """Selects the best strategy for a test case based on Q-table values."""
    if test_case in st.session_state.q_table:
        return max(
            st.session_state.q_table[test_case],
            key=st.session_state.q_table[test_case].get,
            default=None,
        )
    return None  # No data yet


# ---- SIDEBAR PROBLEM DESCRIPTION ----
st.sidebar.header("📖 Problem Description")

st.sidebar.markdown(
    """
    ### **Finding the Second Largest Element**
    Given a list of integers, our goal is to **find the second largest distinct number**.  
    - If there's no second distinct number, return `-1`.
    - Handle edge cases such as duplicate numbers and small lists.
    - AI will generate and evaluate multiple strategies.
    """
)

# ---- SIDEBAR USER CONTROLS ----
st.sidebar.header("⚙️ Select Training Parameters")


# User chooses how many strategies to generate
num_strategies = st.sidebar.slider(
    "🛠️ Number of Strategies", 1, len(actions), len(actions)
)

# User chooses how many test cases to evaluate
num_scenarios = st.sidebar.slider(
    "🔢 Number of Test Scenarios", 1, len(TEST_CASES), len(TEST_CASES)
)

# Dynamically filter based on user input
selected_actions = actions[:num_strategies]
selected_test_cases = TEST_CASES[:num_scenarios]

# ---- MAIN INTERFACE ----

# ---- DISPLAY SELECTED TEST CASES ----
st.markdown("### 🧪 Test Scenarios")
with st.expander("📌 Selected Test Cases", expanded=True):
    for i, (array, expected_output) in enumerate(selected_test_cases):
        st.markdown(f"**Test {i+1}:** `{array}` → Expected Output: `{expected_output}`")

# ---- DISPLAY SELECTED STRATEGIES ----
with st.expander("🛠 Selected AI Strategies", expanded=True):
    st.markdown("Here are the AI strategies used for solving the problem:")
    for action in selected_actions:
        st.markdown(f"- **{action}**")


# ---- DISPLAY THE EVALUATION FUNCTION ----
with st.expander("🔎 View Evaluation Function", expanded=False):
    st.markdown("### 🧠 How AI Solutions Are Evaluated")
    st.code(
        """
def evaluate_solution(solution_code, array, expected_output):
    \"\"\"Runs and evaluates the AI-generated solution.\"\"\"
    if not solution_code:
        return 0

    try:
        local_scope = {}
        exec(solution_code, globals(), local_scope)
        second_largest = local_scope.get("second_largest")

        if not second_largest:
            raise ValueError("Function 'second_largest' not found in generated code")

        tracemalloc.start()

        start_time = time.time()
        result = second_largest(array)
        execution_time = time.time() - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        accuracy = 1.0 if result == expected_output else 0
        efficiency_score = 1 / (execution_time + 1e-6)
        space_score = 1 / (peak + 1e-6)

        reward = (accuracy * 10) + (efficiency_score * 100) + (space_score * 50)
        
        return {
            "Accuracy": accuracy,
            "Execution Time": execution_time,
            "Memory Used": peak / 1024,  # Convert to KB
            "Reward": reward
        }
    except Exception as e:
        return {
            "Accuracy": 0,
            "Execution Time": "Error",
            "Memory Used": "Error",
            "Reward": -10,
            "Error": str(e),
        }
        """,
        language="python",
    )

with st.expander(f"🤖 AI Prompt"):
    st.markdown(
        """
                Write a Python function named `second_largest` that finds the second largest distinct element in an array.
                Use this approach: $_ACTION_. Ensure the function handles edge cases properly.
                Only return the function implementation, no explanations or test cases.
        """
    )

if st.button("📜 Generate Code Functions using AI", key="generate"):
    st.markdown("### 📝 AI-Generated Solutions")
    st.session_state.solutions = generate_solutions()

if st.button("🚀 Run Evaluations", key="evaluate"):
    if not st.session_state.solutions:
        st.warning("Please generate solutions first!")
    else:
        st.markdown("### 🔄 AI Learning Process")
        rewards = {
            action: [] for action in selected_actions
        }  # Use only selected strategies

        for i, (array, expected_output) in enumerate(
            selected_test_cases
        ):  # Use only selected test cases
            with st.expander(f"🧪 Test Case {i+1}: `{array}`"):
                st.markdown(f"✅ Expected Output: `{expected_output}`")

                for action, solution_code in st.session_state.solutions.items():
                    if not solution_code:
                        st.markdown(f"❌ No solution generated for `{action}`.")
                        continue

                    reward = evaluate_solution(solution_code, array, expected_output, action)  # ✅ CALLING evaluate_solution()

                    # Store reward
                    rewards[action].append(reward)

                    # Display the evaluation results
                    st.markdown(
                        f"""
                        **🛠 Strategy:** `{action}`
                        ✅ **Accuracy:** `{1.0 if reward > 0 else 0:.2f}`
                        ⏱ **Execution Time:** `Check Q-table`
                        💾 **Memory Used:** `Check Q-table`
                        🏅 **Reward:** `{reward:.2f}`
                        """,
                        unsafe_allow_html=True,
                    )


                    time.sleep(0.5)

    # ---- PLOT PERFORMANCE ----
    fig, ax = plt.subplots()
    for action, reward_values in rewards.items():
        ax.plot(
            range(1, len(selected_test_cases) + 1),
            reward_values,
            marker="o",
            linestyle="-",
            label=action,
        )

    ax.set_xlabel("Test Cases")
    ax.set_ylabel("Reward Score")
    ax.set_title("📊 AI Strategy Performance Across Test Cases")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # ---- SELECT BEST STRATEGY ----
    avg_rewards = {
        action: np.mean(reward_values) for action, reward_values in rewards.items()
    }
    best_strategy = max(avg_rewards, key=avg_rewards.get)

    st.markdown(f"## 🏆 Best Strategy Selected: `{best_strategy}`")
    st.markdown(f"### 🎯 Average Reward: `{avg_rewards[best_strategy]:.2f}`")

with st.expander("📊 Q-Table (AI Learning Progress)", expanded=True):
    st.markdown("### 🔎 Reinforcement Learning Q-Table")
    for test_case, strategies in st.session_state.q_table.items():
        st.markdown(f"**Test Case:** `{test_case}`")
        for action, q_value in strategies.items():
            st.markdown(f"- **{action}** → Q-Value: `{q_value:.2f}`")
        st.markdown("---")  # Separator for readability
