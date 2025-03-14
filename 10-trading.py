import streamlit as st
import pandas as pd
import numpy as np
import random
import time    
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(page_title="AI-Optimized Order Matching", layout="wide")

st.title("🌐 AI-Optimized Order Matching")

# ------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# ------------------------------------------------------------------------
st.sidebar.header("🔧 Controls")
n_orders = st.sidebar.slider("Orders to Generate", 10, 2000, 100)
seed_val = st.sidebar.number_input("Random Seed", 1, 9999, 42)
label_noise = st.sidebar.slider("Label Noise Fraction", 0.0, 0.5, 0.1)
threshold_ms = st.sidebar.slider("Fast-Match Threshold (ms)", 1, 100, 10)

st.sidebar.subheader("New Order Generation")
num_new_orders = st.sidebar.slider("Number of new orders", 10, 1000, 50)
seed_new = st.sidebar.number_input("New Data Seed", 1, 9999, 123)

st.sidebar.subheader("⚙️ Simulation Settings")
exec_speed = st.sidebar.slider("Execution Speed (ms per step)", 10, 500, 100)
exec_mode = st.sidebar.radio("Simulation Mode", ["Sequential", "Batch"], index=0)
batch_size = st.sidebar.slider("Batch Size (if Batch Mode)", 1, 20, 5)

# Global Ticker Options
TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "META", "NVDA", "NFLX"]

# ------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🏠 Home",
        "📦 Generate Historical Data",
        "🛠 Train AI Model",
        "🔮 Predict & Prioritize Orders",
        "🚀 Run Trading Simulation",
    ]
)

# ------------------------------------------------------------------------
# 0) HELPER FUNCTIONS & DATA GENERATION
# ------------------------------------------------------------------------
def generate_labeled_orders(
    n_orders=100, seed_val=42, label_noise=0.1, threshold_ms=10
):
    """
    Example approach:
      1) Each order has an actual arrival_time.
      2) We pass it to a mini limit-order-book engine *at* that arrival_time.
      3) If it gets matched instantly (or within a short delay), we record matched_time.
      4) time_to_match = matched_time - arrival_time.

    'fast_match=1' if time_to_match < some threshold (like 10 ms).
    """

    np.random.seed(seed_val)
    random.seed(seed_val)

    # 1) Generate basic order attributes
    order_ids = np.arange(n_orders)
    sides = np.random.choice(["BUY", "SELL"], size=n_orders)
    tickers = np.random.choice(TICKERS, size=n_orders)
    prices = np.random.normal(100, 15, size=n_orders).clip(1).round(2)
    sizes = np.random.randint(1, 1000, size=n_orders)

    # 2) Simulate arrival times in ascending order for realism
    #    e.g., first order arrives at base_time, subsequent orders arrive a few ms or ms apart
    base_time = datetime.now()
    # We'll space arrivals randomly between 0..5 ms increments
    # This is just an example; you can use bigger or smaller intervals.
    arrival_offsets = np.cumsum(np.random.randint(0, 5, n_orders))
    arrival_times = [
        base_time + timedelta(milliseconds=int(off)) for off in arrival_offsets
    ]

    # 3) We'll store the matched_time for each order, or None if unmatched
    matched_times = [None] * n_orders

    # Book structures: (price, size, order_id, maybe arrival_time)
    buy_book = []
    sell_book = []

    # Let's define a small function that tries to match upon arrival
    def try_match(side, price, size, oid, arrival_t):
        nonlocal buy_book, sell_book
        # We'll assume if it can match, it does so "instantaneously" at arrival_t
        # or with a tiny constant processing_delay if you want
        processing_delay_ms = 1
        matched_t = arrival_t  # base

        if side == "BUY":
            while size > 0 and sell_book and price >= sell_book[0][0]:
                best_sell_price, best_sell_size, best_sell_oid, best_sell_time = (
                    sell_book[0]
                )
                fill_sz = min(size, best_sell_size)
                size -= fill_sz
                best_sell_size -= fill_sz

                # Record matched_time for both if not matched yet
                if matched_times[oid] is None:
                    # add a small processing delay
                    matched_t = arrival_t + timedelta(milliseconds=processing_delay_ms)
                    matched_times[oid] = matched_t
                if matched_times[best_sell_oid] is None:
                    matched_times[best_sell_oid] = matched_t

                if best_sell_size <= 0:
                    sell_book.pop(0)
                else:
                    sell_book[0] = (
                        best_sell_price,
                        best_sell_size,
                        best_sell_oid,
                        best_sell_time,
                    )
                    break
            # leftover goes to buy_book
            if size > 0:
                buy_book.append((price, size, oid, arrival_t))
                # keep buy_book sorted descending by price
                buy_book.sort(key=lambda x: x[0], reverse=True)
        else:  # SELL
            while size > 0 and buy_book and price <= buy_book[0][0]:
                best_buy_price, best_buy_size, best_buy_oid, best_buy_time = buy_book[0]
                fill_sz = min(size, best_buy_size)
                size -= fill_sz
                best_buy_size -= fill_sz

                if matched_times[oid] is None:
                    matched_t = arrival_t + timedelta(milliseconds=processing_delay_ms)
                    matched_times[oid] = matched_t
                if matched_times[best_buy_oid] is None:
                    matched_times[best_buy_oid] = matched_t

                if best_buy_size <= 0:
                    buy_book.pop(0)
                else:
                    buy_book[0] = (
                        best_buy_price,
                        best_buy_size,
                        best_buy_oid,
                        best_buy_time,
                    )
                    break
            if size > 0:
                sell_book.append((price, size, oid, arrival_t))
                sell_book.sort(key=lambda x: x[0])

    # 4) Process each order in ascending arrival_time
    #    i.e., we iterate from the earliest arrival to the latest.
    #    The ith order is the earliest, then i+1, etc.
    #    We call try_match(...) for each newly arrived order
    sorted_indices = np.argsort(arrival_times)
    for i in sorted_indices:
        side = sides[i]
        price = prices[i]
        size_ = sizes[i]
        arr_t = arrival_times[i]
        try_match(side, price, size_, i, arr_t)

    # 5) Now compute time_to_match in ms
    time_to_match_ms = []
    for i in range(n_orders):
        if matched_times[i] is None:
            time_to_match_ms.append(-1)
        else:
            diff = (matched_times[i] - arrival_times[i]).total_seconds() * 1000
            time_to_match_ms.append(diff)

    # 6) Define a threshold for 'fast match' e.g. 10 ms
    fast_match = np.zeros(n_orders, dtype=int)
    threshold_ms = threshold_ms
    for i in range(n_orders):
        if time_to_match_ms[i] != -1 and time_to_match_ms[i] < threshold_ms:
            fast_match[i] = 1

    # add label noise if desired
    noise_idx = np.random.choice(
        n_orders, size=int(label_noise * n_orders), replace=False
    )
    fast_match[noise_idx] = 1 - fast_match[noise_idx]

    # 7) Build the final DataFrame
    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "arrival_time": arrival_times,
            "side": sides,
            "ticker": tickers,
            "price": prices,
            "size": sizes,
            "matched_time": matched_times,
            "time_to_match_ms": time_to_match_ms,
            "fast_match": fast_match,
        }
    )

    return df


# ------------------------------------------------------------------------
# 1) HOME
# ------------------------------------------------------------------------
with tabs[0]:
    st.header("Welcome & Key Concepts")
    st.markdown(
        """
    **Key Points**:
    - This demo uses **synthetic** data but aims to show how:
      1. We train an ML model (Random Forest) to predict 'fast_match' (did an order match quickly?).
      2. We add a **latency** feature to each order – it's not a sign the order is matched, 
         but rather a random "time cost" of that order.
      3. We then generate *new* orders, predict their 'match_probability', 
         and see how **AI-based prioritization** (sorting by probability) 
         compares to **FIFO** in a toy limit order book simulation.
    """
    )


# ------------------------------------------------------------------------
# 2) GENERATE DATA
# ------------------------------------------------------------------------
with tabs[1]:

    if "training_data" not in st.session_state:
        st.session_state["training_data"] = None  # Ensure key exists

    # Ensure df_train always refers to session state data if it exists
    if st.session_state["training_data"] is not None:
        df_train = st.session_state["training_data"]
        with st.expander("**🔢 Historical data**"):
            st.dataframe(df_train)
    else:
        df_train = pd.DataFrame()  # Default empty DataFrame

    st.header("📦 Generate Historical Data")

    # match_time_threshold = st.sidebar.slider("Match Time Threshold", 1, 100, 10)
    threshold_ms = st.slider("Fast-match threshold (ms)", 1, 100, 10)

    if st.button("Generate Historical Data"):
        df_train = generate_labeled_orders(
            n_orders=n_orders,
            seed_val=seed_val,
            label_noise=label_noise,
            threshold_ms=threshold_ms,
        )
        st.session_state["training_data"] = df_train

        # Convert milliseconds to formatted time (HH:MM:SS.sss)
        df_train["arrival_time"] = pd.to_datetime(df_train["arrival_time"])
        df_train["matched_time"] = pd.to_datetime(df_train["matched_time"], errors="coerce")

        df_train["time_to_match_ms"] = df_train["time_to_match_ms"].apply(
            lambda x: f"{x/1000:.3f} sec" if x >= 0 else "Unmatched"
        )

        df_filtered = df_train[df_train["time_to_match_ms"] != "Unmatched"].copy()
        df_filtered["time_to_match_ms"] = (
            df_filtered["time_to_match_ms"].str.replace(" sec", "").astype(float)
        )

        with st.expander("**🔢 Historical data**"):
            st.dataframe(df_train)

        st.subheader("📈 Summary Statistics")

        # Define column layout for better readability
        col1, col2, col3 = st.columns(3)

        # Compute required statistics
        num_orders = len(df_train)
        num_unmatched = df_train['time_to_match_ms'].eq("Unmatched").sum()
        fast_match_rate = df_train['fast_match'].mean() * 100
        median_time_to_match = df_filtered['time_to_match_ms'].median()
        avg_time_to_match = df_filtered['time_to_match_ms'].mean()
        max_time_to_match = df_filtered['time_to_match_ms'].max()
        min_time_to_match = df_filtered['time_to_match_ms'].min()
        avg_order_price = df_train["price"].mean()
        avg_order_size = df_train["size"].mean()

        # Display statistics in columns
        with col1:
            st.metric(label="📊 Total Orders", value=num_orders)
            st.metric(label="📉 Unmatched Orders", value=num_unmatched)
            st.metric(label="⏳ Min Time to Match (sec)", value=f"{min_time_to_match:.3f}")

        with col2:
            st.metric(label="⚡ Fast Match Rate (%)", value=f"{fast_match_rate:.2f}%")
            st.metric(label="📈 Median Time to Match (sec)", value=f"{median_time_to_match:.3f}")
            st.metric(label="⏱️ Max Time to Match (sec)", value=f"{max_time_to_match:.3f}")

        with col3:
            st.metric(label="💰 Average Order Price ($)", value=f"{avg_order_price:.2f}")
            st.metric(label="📦 Average Order Size", value=f"{avg_order_size:.1f}")
            st.metric(label="⌛ Average Time to Match (sec)", value=f"{avg_time_to_match:.3f}")

        st.subheader("📊 Price vs. Time to Match")

        # Convert 'time_to_match_ms' to numerical format, excluding unmatched values
        df_filtered = df_train[df_train["time_to_match_ms"] != "Unmatched"].copy()
        df_filtered["time_to_match_ms"] = df_filtered["time_to_match_ms"].str.replace(" sec", "").astype(float)

        # Create the scatter plot with a better color scale
        fig_pattern = px.scatter(
            df_filtered,
            x="price",
            y="time_to_match_ms",
            color="fast_match",
            hover_data=["order_id", "side", "size", "ticker"],
            title="Historical Data: Price vs. Time to Match",
            labels={"time_to_match_ms": "Time to Match (Seconds)"},
            color_continuous_scale="viridis",  # Change color scale (options: "viridis", "plasma", "cividis", "magma", "coolwarm")
        )

        # Improve marker visibility
        fig_pattern.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='black')))

        # Adjust layout for better readability
        fig_pattern.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Time to Match (Seconds)",
            coloraxis_colorbar=dict(title="Fast Match Probability"),
            template="plotly_white"
        )

        st.plotly_chart(fig_pattern, use_container_width=True)

    else:
        st.info("Click 'Generate Historical Data' to create a new dataset.")

# ------------------------------------------------------------------------
# 3) TRAIN AI MODEL
# ------------------------------------------------------------------------
with tabs[2]:
    st.header("🛠 Train AI Model")

    if (
        "training_data" not in st.session_state
        or st.session_state["training_data"] is None
    ):
        st.error(
            "No historical data found. Please generate data in 'Generate Historical Data' tab first."
        )
        st.stop()

    df_train = st.session_state["training_data"]
    with st.expander("**🔢 Historical data**"):
        st.dataframe(df_train)

    if st.button("Train RandomForest model"):
        # Show initial message
        st.write("**Training RandomForest model...**")

        # Convert time_to_match_ms to numeric
        df_train["time_to_match_ms"] = df_train["time_to_match_ms"].apply(
            lambda x: (
                np.log1p(pd.to_timedelta(x).total_seconds() * 1000)
                if x != "Unmatched"
                else np.log1p(5000)
            )
        )

        # Encode categorical features
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_features = encoder.fit_transform(df_train[["side", "ticker"]])
        encoded_df = pd.DataFrame(
            encoded_features, columns=encoder.get_feature_names_out(["side", "ticker"])
        )
        df_train = df_train.join(encoded_df).drop(["side", "ticker"], axis=1)

        with st.expander("**🔢 Training Data Overview**"):
            feature_cols = [col for col in df_train.columns if col not in ["fast_match", "order_id", "arrival_time", "matched_time"]]
            target_col = "fast_match"

            # Create a copy of the dataset
            df_display = df_train.copy()

            # Rename columns with emojis
            renamed_columns = {
                col: f"🔹 {col}" if col in feature_cols else
                    f"🎯 {col}" if col == target_col else
                    f"❌ {col}"
                for col in df_train.columns
            }

            df_display = df_display.rename(columns=renamed_columns)

            # Display the modified DataFrame with renamed columns
            st.dataframe(df_display)

        # Then define X, y
        drop_cols = ["order_id", "arrival_time", "matched_time", "fast_match"]
        X = df_train.drop(columns=drop_cols, errors="ignore")
        y = df_train["fast_match"]

        # 🟢 Store the training column names for future re-indexing
        st.session_state["train_columns"] = list(X.columns)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Standardize numeric features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        start_time = time.time()
        st.spinner("Training in progress...")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_final = RandomForestClassifier(n_estimators=100, random_state=42)

        accuracies, precisions, recalls, f1s = [], [], [], []

        progress_bar = st.progress(0)  # Initialize progress bar
        total_folds = skf.get_n_splits()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_s, y_train), start=1):
            model_final.fit(X_train_s[train_idx], y_train.iloc[train_idx])
            y_pred = model_final.predict(X_train_s[val_idx])

            accuracies.append(accuracy_score(y_train.iloc[val_idx], y_pred))
            precisions.append(precision_score(y_train.iloc[val_idx], y_pred))
            recalls.append(recall_score(y_train.iloc[val_idx], y_pred))
            f1s.append(f1_score(y_train.iloc[val_idx], y_pred))

            progress_bar.progress(fold / total_folds)  # Update progress

        progress_bar.empty()  # Remove progress bar after completion

        # Store final trained model
        st.session_state["trained_model"] = (model_final, scaler)

        # Show final training time
        end_time = time.time()
        training_duration = end_time - start_time
        st.success(f"**Final model training complete in {training_duration:.2f} seconds!**")

        # Summary stats
        st.subheader("📈 Model Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ Accuracy", f"{np.mean(accuracies) * 100:.2f}%")
        col2.metric("🎯 Precision", f"{np.mean(precisions) * 100:.2f}%")
        col3.metric("🔄 Recall", f"{np.mean(recalls) * 100:.2f}%")
        col4.metric("📊 F1 Score", f"{np.mean(f1s) * 100:.2f}%")

        # Show iteration chart
        st.subheader("📉 Training Progress Chart")
        df_iters = pd.DataFrame(
            {"Iteration": range(1, len(accuracies) + 1), "Test Accuracy": accuracies}
        )
        fig_iters = px.line(
            df_iters,
            x="Iteration",
            y="Test Accuracy",
            markers=True,
            title="Test Accuracy Over Cross-Validation Iterations",
        )
        st.plotly_chart(fig_iters, use_container_width=True)

        # Feature Importance Chart
        st.subheader("📊 Feature Importance")
        feature_importances = pd.DataFrame(
            {"Feature": X.columns, "Importance": model_final.feature_importances_}
        ).sort_values(by="Importance", ascending=False)

        fig_importance = px.bar(
            feature_importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance in Model",
        )
        st.plotly_chart(fig_importance, use_container_width=True)

# ------------------------------------------------------------------------
# 4) PREDICT & PRIORITIZE ORDERS
# ------------------------------------------------------------------------
with tabs[3]:
    st.header("🔮 Predict & Prioritize Orders")

    # Make sure the model is trained
    if (
        "trained_model" not in st.session_state
        or st.session_state["trained_model"] is None
    ):
        st.error("No trained model found. Train the model first in '🛠 Train AI Model'.")
        st.stop()

    model_final, scaler = st.session_state["trained_model"]

    # Ensure we have the training columns
    if (
        "train_columns" not in st.session_state
        or st.session_state["train_columns"] is None
    ):
        st.error("No training columns found. Train the model first.")
        st.stop()

    train_cols = st.session_state["train_columns"]

    if "df_new" not in st.session_state:
        st.session_state["df_new"] = None

    # Generate new unlabeled orders
    if st.button("Generate New Orders"):
        np.random.seed(seed_new)
        random.seed(seed_new)

        order_ids = np.arange(num_new_orders)
        sides = np.random.choice(["BUY", "SELL"], size=num_new_orders)
        tickers = np.random.choice(TICKERS, size=num_new_orders)
        prices = np.random.normal(100, 15, size=num_new_orders).clip(1).round(2)
        sizes = np.random.randint(1, 1000, size=num_new_orders)
        base_time = datetime.now()
        arrival_offsets = np.cumsum(np.random.randint(0, 5, num_new_orders))
        arrival_times = [
            base_time + timedelta(milliseconds=int(off)) for off in arrival_offsets
        ]

        df_new = pd.DataFrame(
            {
                "order_id": order_ids,
                "side": sides,
                "ticker": tickers,
                "price": prices,
                "size": sizes,
                "arrival_time": arrival_times,
            }
        )

        st.session_state["df_new"] = df_new
        st.success("New unlabeled orders generated successfully!")

    # If we have df_new, let's show and do "Predict & Prioritize"
    if st.session_state["df_new"] is not None:
        df_new = st.session_state["df_new"]
        st.write("**Preview of newly generated unlabeled orders:**")
        st.dataframe(df_new.head(20))

        if st.button("Predict & Prioritize"):
            # We'll do the same categorical encoding approach
            df_infer = df_new.copy()

            # Example: convert side/ticker via get_dummies
            df_infer = pd.get_dummies(df_infer, columns=["side", "ticker"])

            # Remove columns not used in training
            drop_cols = ["order_id", "arrival_time"]
            for c in drop_cols:
                if c in df_infer.columns:
                    df_infer.drop(c, axis=1, inplace=True)

            # Reindex to match the columns used during training
            df_infer = df_infer.reindex(columns=train_cols, fill_value=0)

            X_infer_s = scaler.transform(df_infer)  # scale
            match_probs = model_final.predict_proba(X_infer_s)[:, 1]
            df_new["match_probability"] = match_probs

            # Sort descending by match_probability => AI approach
            df_ai_sorted = df_new.sort_values("match_probability", ascending=False)
            # Also define FIFO approach => sorted by arrival_time
            df_fifo_sorted = df_new.sort_values("arrival_time")

            st.markdown("### AI-Prioritized Orders (Highest Probability First)")
            st.dataframe(df_ai_sorted.head(20))

            # Store them for next tab
            st.session_state["df_fifo"] = df_fifo_sorted
            st.session_state["df_ai"] = df_ai_sorted
            st.success(
                "Prioritization complete. Proceed to '🚀 Run Trading Simulation'!"
            )
    else:
        st.info("Click 'Generate New Orders' to create a new unlabeled dataset.")

with tabs[4]:
    st.header("🚀 AI vs. FIFO Trading Simulation")

    # Ensure necessary prior data exists
    if "df_fifo" not in st.session_state or "df_ai" not in st.session_state:
        st.error(
            "No prioritized orders found. Run '🔮 Predict & Prioritize Orders' first."
        )
        st.stop()

    # Load sorted order books
    fifo_orders = st.session_state["df_fifo"].copy()
    ai_orders = st.session_state["df_ai"].copy()

    # Initialize session state for execution tracking
    if "fifo_executed" not in st.session_state:
        st.session_state["fifo_executed"] = []
        st.session_state["ai_executed"] = []

    
    # Execution Logic
    def execute_orders(order_list, mode="FIFO"):
        executed_orders = []
        timestamps = []
        base_time = datetime.now()

        for idx, (_, order) in enumerate(order_list.iterrows()):
            time.sleep(exec_speed / 1000.0)  # Simulate latency
            execution_time = base_time + timedelta(milliseconds=idx * exec_speed)

            # Ensure non-negative execution time
            order["execution_time"] = execution_time
            executed_orders.append(order.to_dict())
            timestamps.append(execution_time)

            # Batch Processing
            if mode == "Batch" and (idx + 1) % batch_size == 0:
                time.sleep(exec_speed / 1000.0)
                st.write(f"🔄 Processed batch of {batch_size} orders...")

        return executed_orders, timestamps

    # Run Simulation
    if st.button("▶ Run Trading Simulation"):
        with st.spinner("Executing FIFO orders..."):
            st.session_state["fifo_executed"], fifo_times = execute_orders(
                fifo_orders, exec_mode
            )

        with st.spinner("Executing AI-prioritized orders..."):
            st.session_state["ai_executed"], ai_times = execute_orders(
                ai_orders, exec_mode
            )

        st.success("✅ Simulation Completed Successfully!")

        # Convert executed orders to DataFrame
        df_fifo_exec = pd.DataFrame(st.session_state["fifo_executed"])
        df_ai_exec = pd.DataFrame(st.session_state["ai_executed"])

        # Execution Time Calculation (Fix Negative Times)
        fifo_avg_time = (
            np.mean([(t - fifo_times[0]).total_seconds() for t in fifo_times])
            if fifo_times
            else 0
        )
        ai_avg_time = (
            np.mean([(t - ai_times[0]).total_seconds() for t in ai_times])
            if ai_times
            else 0
        )

        # Ensure non-negative times
        fifo_avg_time = max(fifo_avg_time, 0)
        ai_avg_time = max(ai_avg_time, 0)

        # 📊 Execution Performance Metrics
        st.subheader("📊 Execution Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("FIFO Avg Execution Time (sec)", f"{fifo_avg_time:.3f}")

        with col2:
            st.metric("AI Avg Execution Time (sec)", f"{ai_avg_time:.3f}")

        # 📈 Execution Timeline Chart (Fixed `range()` issue)
        st.subheader("📈 Execution Timeline")
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fifo_times,
                y=list(range(len(fifo_times))),  # ✅ FIXED: Converted range to list
                mode="lines",
                name="FIFO Execution",
                line=dict(width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ai_times,
                y=list(range(len(ai_times))),
                mode="lines",
                name="AI Execution",
                line=dict(width=3, dash="dot"),
            )
        )

        fig.update_layout(
            title="Order Execution Over Time",
            xaxis_title="Time",
            yaxis_title="Executed Order Count",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # 📋 Final Executed Orders Table
        st.subheader("📋 Final Executed Orders")
        st.write("### FIFO Execution")
        st.dataframe(
            df_fifo_exec.style.format(
                {"execution_time": lambda t: t.strftime("%H:%M:%S.%f")}
            )
        )
        st.write("### AI Execution")
        st.dataframe(
            df_ai_exec.style.format(
                {"execution_time": lambda t: t.strftime("%H:%M:%S.%f")}
            )
        )
