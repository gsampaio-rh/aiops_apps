import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------
st.set_page_config(page_title="AI-Optimized Order Matching (Enhanced)", layout="wide")

st.title("🌐 AI-Optimized Order Matching – Enhanced Demo")

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
        "👓 Meeting with Steve",
    ]
)

# Global Ticker Options
TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "META", "NVDA", "NFLX"]


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
    st.header("📦 Generate Historical Data")

    st.sidebar.header("Data Generation Controls")
    n_orders = st.sidebar.slider("Number of orders to generate", 10, 2000, 100)
    seed_val = st.sidebar.number_input("Random seed", 1, 9999, 42)
    label_noise = st.sidebar.slider("Label Noise fraction", 0.0, 0.5, 0.1)

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
        df_train["time_to_match_ms"] = df_train["time_to_match_ms"].apply(
            lambda x: (str(timedelta(milliseconds=x)) if x >= 0 else "Unmatched")
        )

        st.write("**Historical data (head)**:")
        st.dataframe(df_train)

        # Optional: Show any data pattern or threshold-based charts here
        st.subheader("Visualizing Data Patterns")
        fig_pattern = px.scatter(
            df_train,
            x="price",
            y="time_to_match_ms",
            color="fast_match",
            hover_data=["order_id", "side", "size", "ticker"],
            title="Historical Data: Price vs. time_to_match_ms",
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

    st.write("**Using the existing historical dataset**:")
    st.dataframe(df_train.head(10))

    # If needed, do dummies etc.
    st.write("**Training RandomForest model...**")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Convert time_to_match_ms to numeric (keeping unmatched as -1)
    df_train["time_to_match_ms"] = df_train["time_to_match_ms"].apply(
        lambda x: pd.to_timedelta(x).total_seconds() * 1000 if x != "Unmatched" else -1
    )

    # Re-encode categorical columns
    df_enc = pd.get_dummies(df_train, columns=["side", "ticker"])

    # Drop unnecessary columns (including datetime ones)
    X = df_enc.drop(["order_id", "fast_match", "arrival_time", "matched_time"], axis=1)
    y = df_enc["fast_match"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # <-- No more conversion errors
    X_test_s = scaler.transform(X_test)

    n_iters = 10
    iteration_accuracies = []
    model_final = None

    for i in range(n_iters):
        subset_indices = np.random.choice(
            len(X_train_s), size=int(0.8 * len(X_train_s)), replace=False
        )
        X_sub = X_train_s[subset_indices]
        y_sub = y_train.iloc[subset_indices]

        model = RandomForestClassifier(n_estimators=50, random_state=(42 + i))
        model.fit(X_sub, y_sub)

        test_acc = model.score(X_test_s, y_test)
        iteration_accuracies.append(test_acc)

        progress_bar.progress(int(100 * (i + 1) / n_iters))
        status_text.text(
            f"Iteration {i+1}/{n_iters} - Test Accuracy: {test_acc*100:.2f}%"
        )
        time.sleep(0.2)

        model_final = model

    # Store final
    st.session_state["trained_model"] = (model_final, scaler)

    st.write("**Final model training complete**")

    # Show iteration chart
    st.subheader("Training Progress Chart")
    df_iters = pd.DataFrame(
        {"iteration": range(1, n_iters + 1), "test_acc": iteration_accuracies}
    )
    fig_iters = px.line(
        df_iters,
        x="iteration",
        y="test_acc",
        markers=True,
        title="Test Accuracy Over Iterations",
    )
    st.plotly_chart(fig_iters, use_container_width=True)
