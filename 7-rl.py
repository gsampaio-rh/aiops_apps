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
import pandas as pd

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
    """Streams AI solutions for selected strategies with real-time status updates."""
    solutions = {}
    total_actions = len(selected_actions)

    progress_bar = st.progress(0)  # Overall progress indicator
    step_progress = 1 / total_actions  # Progress per strategy

    with st.spinner("🔄 Generating AI code functions..."):
        for i, action in enumerate(
            selected_actions
        ):  # Loop through selected strategies
            # ✅ Create a dynamic card for each strategy (always visible)
            status_placeholder = st.empty()
            code_placeholder = st.empty()

            status_placeholder.markdown(
                f"""
                <div style="
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #f5f5f5;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                    font-size: 16px;">
                    ✨ <b>Building Strategy:</b> {action} ... ⏳
                </div>
                """,
                unsafe_allow_html=True,
            )

            prompt = f"""
            Write a Python function named `second_largest` that finds the second largest distinct element in an array.
            Use this approach: {action}. Ensure the function handles edge cases properly.
            Only return the function implementation, no explanations or test cases.
            """

            # Step 1: Show "Fetching AI Response..."
            time.sleep(0.5)  # Small delay for effect
            status_placeholder.markdown(
                f"🚀 **Generating `{action}` function... Please wait!**"
            )

            # Step 2: Stream AI Response (Real-time code display)
            response = ""
            for chunk in llm.stream(prompt):
                response += chunk
                code_placeholder.markdown(
                    f"```python\n{response}\n```"
                )  # Update UI dynamically

            # Step 3: Process & Clean Response
            match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            solutions[action] = match.group(1).strip() if match else response.strip()

            # Step 4: Display Final Code
            code_placeholder.code(solutions[action], language="python")

            # Step 5: Show Success ✅
            status_placeholder.markdown(
                f"""
                <div style="
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #e3f2fd;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                    font-size: 16px;">
                    ✅ <b>Completed:</b> {action} function is ready! 🎉
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Update Progress Bar
            progress_bar.progress(min((i + 1) * step_progress, 1.0))

    return solutions


def evaluate_solution(solution_code, array, expected_output, action):
    """Runs and evaluates the AI-generated solution and updates Q-table."""
    if not solution_code:
        update_q_table(tuple(array), action, -10)  # Penalize missing solutions
        return {"reward": -10, "execution_time": "N/A", "memory_usage": "N/A"}

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

        return {
            "reward": reward,
            "execution_time": execution_time,
            "memory_usage": peak / 1024,  # Convert to KB
        }

    except Exception as e:
        update_q_table(tuple(array), action, -10)  # Penalize errors
        return {
            "reward": -10,
            "execution_time": "Error",
            "memory_usage": "Error",
            "error": str(e),
        }


def select_best_strategy(test_case):
    """Selects the best strategy for a test case based on Q-table values."""
    if test_case in st.session_state.q_table:
        return max(
            st.session_state.q_table[test_case],
            key=st.session_state.q_table[test_case].get,
            default=None,
        )
    return None  # No data yet


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


# ---- LLM SETUP ----
llm = Ollama(model="mistral")

# ---- APP CONFIG ----
st.set_page_config(page_title="AI Learning: Reinforcement Training", layout="wide")

# ---- CUSTOM STYLES ----
st.markdown(
    """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #1d1d1f; }
        .header { text-align: center; font-size: 2em; font-weight: bold; padding: 10px; }
        .card { background: #fff; padding: 15px; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
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

# ---- TABBED LAYOUT ----
tabs = st.tabs(
    [
        "Context",
        "Strategies",
        "Reward",
        "Scenarios",
        "Generate Code",
        "Evaluate",
        "Q-Table ",
    ]
)

# ---- CONTEXT TAB ----
with tabs[0]:
    st.markdown(
        """
        ## 🌎 Context
        """
    )
    st.markdown(
        """
            AI models often need to select the best strategy for solving a problem. **Reinforcement Learning** helps models improve by rewarding good decisions and penalizing bad ones. 
            In this demonstration, we use **Q-learning** to refine AI's ability to pick the best approach for finding the second largest element in an array.
        """
    )
    st.markdown(
        """
        ## 📖 The Problem
        """
    )
    st.markdown(
        """
            **Finding the Second Largest Element**
            - Given a list of integers, our goal is to **find the second largest distinct number**.
            - If there's no second distinct number, return `-1`.
            - AI will generate and evaluate multiple strategies to solve this problem efficiently.
            
            ### 🔍 **Example 1:**
            - **Input:** `[10, 20, 20, 5, 8]`
            - **Process:**
            - The largest number is `20`.
            - The second largest distinct number is `10`.
            - **Output:** `10`
            
            ### 🔍 **Example 2:**
            - **Input:** `[5, 5, 5]`
            - **Output:** `-1` (No second distinct number)
            """
    )
    st.markdown(
        """
            ## 🤖 AI Prompt
            """
    )
    st.markdown(
        """
                Write a Python function named `second_largest` that finds the second largest distinct element in an array.
                Use this approach: $_ACTION_. Ensure the function handles edge cases properly.
                Only return the function implementation, no explanations or test cases.
        """
    )


# ---- STRATEGIES TAB ----
with tabs[1]:
    st.markdown("## ⚙️ AI Strategies")
    st.markdown(
        "Explore different approaches to finding the second largest element in a list. Click on a strategy to learn more!"
    )

    strategies = {
        "Sort and Find Second Last": {
            "description": "Sort the list in ascending order and pick the second last unique element.",
            "steps": [
                "Sort the list.",
                "Remove duplicates.",
                "Select the second last element.",
            ],
            "example": {"Input": [3, 1, 4, 1, 5], "Output": 4},
        },
        "Single Pass Scan with Two Variables": {
            "description": "Track the two largest values in a single scan, improving efficiency.",
            "steps": [
                "Initialize two variables: `largest` and `second_largest`.",
                "Iterate through the list, updating them as needed.",
                "Return `second_largest`.",
            ],
            "example": {"Input": [3, 1, 4, 1, 5], "Output": 4},
        },
        "Use Set to Remove Duplicates, Then Find Max": {
            "description": "Remove duplicate values first, then find the second highest.",
            "steps": [
                "Convert the list to a set to remove duplicates.",
                "Sort the unique values.",
                "Return the second last value.",
            ],
            "example": {"Input": [3, 1, 4, 1, 5], "Output": 4},
        },
        "Brute Force Nested Loop Approach": {
            "description": "Compare each element with every other element (inefficient but simple).",
            "steps": [
                "Use two nested loops.",
                "Check for the second largest manually.",
                "Return the result.",
            ],
            "example": {"Input": [3, 1, 4, 1, 5], "Output": 4},
        },
        "Alternative Logic-Based Approach": {
            "description": "A custom AI-designed method.",
            "steps": [
                "Use a unique logic approach.",
                "Optimize based on specific constraints.",
            ],
            "example": {"Input": [3, 1, 4, 1, 5], "Output": 4},
        },
    }

    for strategy, details in strategies.items():
        with st.expander(f"🔹 **{strategy}**"):
            st.write(details["description"])
            st.markdown("**Steps:**")
            for step in details["steps"]:
                st.markdown(f"- {step}")

            st.markdown("**Example:**")
            st.code(
                f"Input: {details['example']['Input']} → Output: {details['example']['Output']}"
            )


# ---- REWARDS TAB ----
with tabs[2]:
    st.markdown("### 🧠 How AI Solutions Are Evaluated")
    with st.expander("evaluate_solution"):
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
    # ---- REWARD FUNCTION ----
    st.markdown("## 🏆 The Reward Function")
    st.markdown(
        """
            The AI receives **rewards or penalties** based on how well a strategy performs. The reward function considers:
            - ✅ **Accuracy** → Whether the strategy finds the correct answer.
            - ⚡ **Efficiency** → Execution speed and memory usage.
            - 🔄 **Adaptability** → Handling edge cases like duplicates and small lists.
            
            The AI adjusts its strategy selection based on past rewards, continuously learning and improving.
            """
    )

# ---- SCENARIOS TAB ----
with tabs[3]:
    # ---- DISPLAY SELECTED TEST CASES ----
    st.markdown("### 🧪 Test Scenarios")
    for i, (array, expected_output) in enumerate(selected_test_cases):
        st.markdown(
            f"- **Test {i+1}:** `{array}` → Expected Output: `{expected_output}`"
        )

# ---- CODE GENERATION TAB ----
with tabs[4]:
    if st.button("📜 Generate Code Functions using AI", key="generate"):
        st.markdown("### 📝 AI-Generated Solutions")
        st.session_state.solutions = generate_solutions()

# ---- RUN EVALUATION TAB ----
with tabs[5]:
    st.markdown("## 🚀 AI Model Evaluation")

    # Run Evaluations Button
    if st.button(
        "🎯 Run Evaluations",
        key="evaluate",
        help="Click to evaluate AI-generated strategies",
    ):

        if not st.session_state.solutions:
            st.warning("⚠️ Please generate solutions first!")
        else:
            st.success("🔄 Evaluation in progress... Please wait.")

            rewards = {
                action: [] for action in selected_actions
            }  # Store rewards per strategy
            evaluation_summary = (
                []
            )  # Store evaluation results for displaying in a DataFrame

            progress_bar = st.progress(0)  # Add a progress bar for better feedback
            total_tests = len(selected_test_cases) * len(selected_actions)
            progress_step = 1 / total_tests

            for i, (array, expected_output) in enumerate(selected_test_cases):
                with st.expander(f"🧪 Test Case {i+1}: `{array}`"):
                    st.markdown(f"✅ **Expected Output:** `{expected_output}`")

                    for action, solution_code in st.session_state.solutions.items():
                        if not solution_code:
                            st.markdown(f"❌ No solution generated for `{action}`.")
                            continue

                        evaluation_results = evaluate_solution(
                            solution_code, array, expected_output, action
                        )

                        reward = evaluation_results["reward"]
                        execution_time = evaluation_results["execution_time"]
                        memory_usage = evaluation_results["memory_usage"]

                        # Color-coded indicators
                        accuracy_status = "🟢 Correct" if reward > 0 else "🔴 Incorrect"

                        # Display results in a card-style format for clarity
                        st.markdown(
                            f"""
                            <div style="
                                padding: 10px; 
                                border-radius: 8px; 
                                background-color: #f8f9fa; 
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1); 
                                margin-bottom: 10px;">
                                <b>🛠 Strategy:</b> {action}<br>
                                {accuracy_status}<br>
                                ⏱ <b>Execution Time:</b> {execution_time if execution_time == 'Error' else f"{float(execution_time):.6f}"} sec<br>
                                💾 <b>Memory Used:</b> {memory_usage if memory_usage == 'Error' else f"{float(memory_usage):.2f}"} KB<br>
                                🏅 <b>Reward:</b> {reward if reward == 'Error' else f"{float(reward):.2f}"}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Store for summary table
                        evaluation_summary.append(
                            {
                                "Test Case": i + 1,
                                "Strategy": action,
                                "Accuracy": "Correct" if reward > 0 else "Incorrect",
                                "Execution Time (sec)": execution_time,
                                "Memory Usage (KB)": memory_usage,
                                "Reward": reward,
                            }
                        )

                        # Store reward data
                        rewards[action].append(reward)

                        # Update progress bar
                        progress_bar.progress(
                            min(
                                (
                                    progress_step
                                    * (
                                        i * len(selected_actions)
                                        + list(st.session_state.solutions.keys()).index(
                                            action
                                        )
                                        + 1
                                    )
                                ),
                                1.0,
                            )
                        )

                        time.sleep(0.5)  # Prevent UI freeze

            # Close progress bar
            progress_bar.empty()

            summary_df = pd.DataFrame(evaluation_summary)

            st.dataframe(summary_df)  # Streamlit's built-in dataframe display

            # ---- PLOT PERFORMANCE ----
            fig, ax = plt.subplots(figsize=(8, 5))
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
            ax.set_title("AI Strategy Performance Across Test Cases")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            # ---- SELECT BEST STRATEGY ----
            avg_rewards = {
                action: np.mean(reward_values)
                for action, reward_values in rewards.items()
            }
            best_strategy = max(avg_rewards, key=avg_rewards.get)

            st.markdown(
                f"""
                <div style="
                    padding: 15px; 
                    border-radius: 8px; 
                    background-color: #e3f2fd; 
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1); 
                    margin-top: 20px;">
                    🎯 <b>Best Strategy Selected:</b> `{best_strategy}`<br>
                    🏆 <b>Average Reward:</b> `{avg_rewards[best_strategy]:.2f}`
                </div>
                """,
                unsafe_allow_html=True,
            )


with tabs[6]:
    with st.expander("📊 Q-Table (AI Learning Progress)", expanded=True):
        st.markdown("### 🔎 Reinforcement Learning Q-Table")

        # Convert Q-Table to a structured DataFrame
        q_table_data = []
        for test_case, strategies in st.session_state.q_table.items():
            for action, q_value in strategies.items():
                q_table_data.append([test_case, action, round(q_value, 2)])

        df = pd.DataFrame(q_table_data, columns=["Test Case", "Action", "Q-Value"])

        # Display table
        st.dataframe(df, use_container_width=True)
