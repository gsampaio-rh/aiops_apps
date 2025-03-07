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
    """Streams AI solutions for finding the second largest number inside expanders."""
    solutions = {}

    for action in actions:
        with st.expander(f"✨ Strategy: {action}"):
            # Create a placeholder inside the expander
            placeholder = st.empty()
            placeholder.markdown(f"⌛ Generating solution for **{action}**...")

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
            solutions[action] = match.group(1).strip() if match else response.strip()

            # Replace placeholder with final code block
            placeholder.code(solutions[action], language="python")

    return solutions


def evaluate_solution(solution_code, array, expected_output):
    """Runs and evaluates the AI-generated solution."""
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
        print(
            f"Accuracy: {accuracy:.2f}, Execution Time: {execution_time:.6f} sec, Memory Used: {peak / 1024:.2f} KB, Reward: {reward:.2f}"
        )

        return reward
    except Exception:
        return -10
# ---- MAIN INTERFACE ----
st.sidebar.header("⚙️ Select Training Parameters")

if st.button("📜 Generate AI Solutions", key="generate"):
    st.markdown("### 📝 AI-Generated Solutions")
    solutions = generate_solutions()

if st.button("🚀 Run Evaluations", key="evaluate"):
    solutions = generate_solutions()
    st.markdown("### 🔄 AI Learning Process")
    rewards = {action: [] for action in actions}

    for i, (array, expected_output) in enumerate(TEST_CASES):
        with st.expander(f"🧪 Test Case {i+1}: `{array}`"):
            st.markdown(f"✅ Expected Output: `{expected_output}`")

            for action, solution_code in solutions.items():
                reward = evaluate_solution(solution_code, array, expected_output)
                rewards[action].append(reward)

                st.markdown(
                    (
                        f"🏅 Reward: <span class='reward-success'>{reward:.2f}</span>"
                        if reward > 0
                        else f"🏅 Reward: <span class='reward-fail'>{reward:.2f}</span>"
                    ),
                    unsafe_allow_html=True,
                )
                time.sleep(0.5)

    # ---- PLOT PERFORMANCE ----
    fig, ax = plt.subplots()
    for action, reward_values in rewards.items():
        ax.plot(
            range(1, len(TEST_CASES) + 1),
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
