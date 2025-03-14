import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
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
        "📁 Train AI Model",
        "🔮 Predict & Prioritize Orders",
        "🚀 Run Trading Simulation",
        "👓 Meeting with Steve",
    ]
)

# ------------------------------------------------------------------------
# UTILS: ORDER BOOK & SYNTHETIC DATA
# ------------------------------------------------------------------------

TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "META", "NVDA", "NFLX"]


def basic_order_book_engine(orders, partial_fill=True):
    """
    A small, simplistic limit order book engine.
    BUY orders: stored descending by price
    SELL orders: stored ascending by price
    If partial_fill=True, an order can match multiple opposing orders.
    Returns a DataFrame of fill events.
    """
    buy_book = []
    sell_book = []
    fills = []

    def insert_buy(b):
        buy_book.append(b)
        buy_book.sort(key=lambda x: x[0], reverse=True)

    def insert_sell(s):
        sell_book.append(s)
        sell_book.sort(key=lambda x: x[0])

    for row in orders.itertuples():
        side = row.side
        price = row.price
        size = row.size
        oid = row.order_id

        if side == "BUY":
            while size > 0 and len(sell_book) > 0 and price >= sell_book[0][0]:
                best_sell_price, best_sell_size, best_sell_oid = sell_book[0]
                fill_sz = min(size, best_sell_size)
                # Fill price is best_sell_price, or some midpoint, but let's keep it simple
                fills.append(
                    {
                        "buy_order_id": oid,
                        "sell_order_id": best_sell_oid,
                        "price": best_sell_price,
                        "size": fill_sz,
                    }
                )
                size -= fill_sz
                best_sell_size -= fill_sz
                if best_sell_size <= 0:
                    sell_book.pop(0)
                else:
                    sell_book[0] = (best_sell_price, best_sell_size, best_sell_oid)
                if not partial_fill:
                    size = 0
                    break
            if size > 0:
                insert_buy((price, size, oid))
        else:  # SELL
            while size > 0 and len(buy_book) > 0 and price <= buy_book[0][0]:
                best_buy_price, best_buy_size, best_buy_oid = buy_book[0]
                fill_sz = min(size, best_buy_size)
                fills.append(
                    {
                        "buy_order_id": best_buy_oid,
                        "sell_order_id": oid,
                        "price": best_buy_price,
                        "size": fill_sz,
                    }
                )
                size -= fill_sz
                best_buy_size -= fill_sz
                if best_buy_size <= 0:
                    buy_book.pop(0)
                else:
                    buy_book[0] = (best_buy_price, best_buy_size, best_buy_oid)
                if not partial_fill:
                    size = 0
                    break
            if size > 0:
                insert_sell((price, size, oid))

    return pd.DataFrame(fills)


def generate_labeled_orders(n_orders, seed_val, label_noise=0.1):
    """
    Generate a 'historical' dataset with labeled 'fast_match' using a toy order book.
    We'll add a random ticker and random latency to each order.
    """
    np.random.seed(seed_val)
    random.seed(seed_val)

    order_ids = np.arange(n_orders)
    sides = np.random.choice(["BUY", "SELL"], size=n_orders)
    tickers = np.random.choice(TICKERS, size=n_orders)
    prices = np.random.normal(100, 15, size=n_orders).clip(1).round(2)
    sizes = np.random.randint(1, 1000, size=n_orders)
    # "latency_ms": how many ms it took to arrive
    latencies = np.random.randint(1, 50, size=n_orders)

    # We'll label "fast_match=1" if it is matched quickly in the next 3 arrivals
    # via a mini simulation (like we did before).
    fill_steps = [-1] * n_orders

    buy_book = []
    sell_book = []

    def match_incoming(new_side, new_price, new_size, new_oid, step_idx):
        nonlocal buy_book, sell_book
        if new_side == "BUY":
            while new_size > 0 and sell_book and (new_price >= sell_book[0][0]):
                sprice, ssize, soid = sell_book[0]
                fill_sz = min(new_size, ssize)
                new_size -= fill_sz
                ssize -= fill_sz
                fill_steps[new_oid] = (
                    step_idx if fill_steps[new_oid] == -1 else fill_steps[new_oid]
                )
                fill_steps[soid] = (
                    step_idx if fill_steps[soid] == -1 else fill_steps[soid]
                )
                if ssize <= 0:
                    sell_book.pop(0)
                else:
                    sell_book[0] = (sprice, ssize, soid)
                    break
            if new_size > 0:
                buy_book.append((new_price, new_size, new_oid))
                buy_book.sort(key=lambda x: x[0], reverse=True)
        else:
            while new_size > 0 and buy_book and (new_price <= buy_book[0][0]):
                bprice, bsize, boid = buy_book[0]
                fill_sz = min(new_size, bsize)
                new_size -= fill_sz
                bsize -= fill_sz
                fill_steps[new_oid] = (
                    step_idx if fill_steps[new_oid] == -1 else fill_steps[new_oid]
                )
                fill_steps[boid] = (
                    step_idx if fill_steps[boid] == -1 else fill_steps[boid]
                )
                if bsize <= 0:
                    buy_book.pop(0)
                else:
                    buy_book[0] = (bprice, bsize, boid)
                    break
            if new_size > 0:
                sell_book.append((new_price, new_size, new_oid))
                sell_book.sort(key=lambda x: x[0])

    for i in range(n_orders):
        match_incoming(sides[i], prices[i], sizes[i], i, i)

    labels = np.zeros(n_orders, dtype=int)
    for i in range(n_orders):
        # "fast" if fill_steps[i] != -1 and fill_steps[i] <= i+3
        if fill_steps[i] != -1 and fill_steps[i] <= (i + 3):
            labels[i] = 1

    # label noise
    noise_idx = np.random.choice(
        n_orders, size=int(label_noise * n_orders), replace=False
    )
    labels[noise_idx] = 1 - labels[noise_idx]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "side": sides,
            "ticker": tickers,
            "price": prices,
            "size": sizes,
            "latency_ms": latencies,
            "fast_match": labels,
        }
    )
    return df


def generate_new_orders(n_orders, seed_val):
    """
    Unlabeled new orders for inference.
    We add random ticker, random latency, etc.
    """
    np.random.seed(seed_val)
    random.seed(seed_val)

    order_ids = np.arange(n_orders)
    sides = np.random.choice(["BUY", "SELL"], size=n_orders)
    tickers = np.random.choice(TICKERS, size=n_orders)
    prices = np.random.normal(100, 15, size=n_orders).clip(1).round(2)
    sizes = np.random.randint(1, 1000, size=n_orders)
    latencies = np.random.randint(1, 50, size=n_orders)

    base_time = datetime.now()
    arrivals = [base_time + pd.Timedelta(i * 10, "ms") for i in range(n_orders)]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "side": sides,
            "ticker": tickers,
            "price": prices,
            "size": sizes,
            "latency_ms": latencies,
            "timestamp": arrivals,
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
    This **enhanced** demo illustrates:
    1. **Stock Names** (tickers) in each order.
    2. **Button & Progress Bar** while training a RandomForest model, with a simple **training process chart**.
    3. **Latency** columns in the data.
    4. A final **'Run Trading Simulation'** button that processes new orders in a small limit order book.
    5. A **2D Scatter Plot** of predicted probabilities.
    6. A short meeting with "Steve Jobs" on how to present this to non-technical people.

    **Flow**:
    - Generate & label “historical” data (tab 2) => Train an ML model **with a progress bar**.
    - Generate new orders => Use the model to predict match probability => Show 2D chart (price vs. size).
    - Click **“Run Trading Simulation”** => We see how AI reordering affects the fill rates & partial fills.
    - Finally, read "Meeting with Steve" for design insight.
    """
    )


# ------------------------------------------------------------------------
# 2) TRAIN AI MODEL
# ------------------------------------------------------------------------
with tabs[1]:
    st.header("📁 Train AI Model on Historical Synthetic Data")

    st.sidebar.header("Training Data Controls")
    n_train = st.sidebar.slider("Number of historical orders", 100, 5000, 1000)
    label_noise = st.sidebar.slider("Label Noise", 0.0, 0.5, 0.1, 0.01)
    seed_val = st.sidebar.number_input("Random Seed (Training)", value=999, step=1)

    # Button to trigger training
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = None
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None

    if st.button("Generate & Train Model"):
        # Generate data
        df_train = generate_labeled_orders(n_train, seed_val, label_noise)

        st.session_state["training_data"] = df_train

        st.write("**Training data sample:**")
        st.dataframe(df_train.head(10))

        # Let’s simulate incremental training with a progress bar
        st.write("**Training RandomForest model...**")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # We'll store a list of "iteration accuracies" to plot
        iteration_accuracies = []

        # Build a random forest classifier with partial incremental approach (mock).
        # Real scikit-learn's RandomForest doesn't partial_fit easily, so we'll
        # simulate it by training multiple sub-models or subsets.
        # Or just do repeated fits with different random subsets.

        # Preprocess
        df_enc = pd.get_dummies(df_train, columns=["side", "ticker"])
        X = df_enc.drop(["order_id", "fast_match"], axis=1)
        y = df_enc["fast_match"]
        # Normal train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # We'll do a loop "n_iters" times, each time training a new random forest on a subset
        n_iters = 10
        model_final = None
        for i in range(n_iters):
            # Subset for training
            subset_indices = np.random.choice(
                len(X_train_s), size=int(0.8 * len(X_train_s)), replace=False
            )
            X_sub = X_train_s[subset_indices]
            y_sub = y_train.iloc[subset_indices]

            model = RandomForestClassifier(n_estimators=50, random_state=(seed_val + i))
            model.fit(X_sub, y_sub)

            train_acc = model.score(X_train_s, y_train)
            test_acc = model.score(X_test_s, y_test)
            iteration_accuracies.append(test_acc)

            progress_bar.progress(int(100 * (i + 1) / n_iters))
            status_text.text(
                f"Iteration {i+1}/{n_iters} - Test Accuracy: {test_acc*100:.2f}%"
            )
            time.sleep(0.2)  # just to show progress

            # We'll keep the last model
            model_final = model

        st.session_state["trained_model"] = (model_final, scaler)

        st.write("**Final model training complete!**")

        # Visualize iteration_accuracies
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

        with st.expander("🔍 How Does Random Forest Training Work?"):
            st.write(
                """
            - We create many **decision trees** on random subsets of the data.
            - Each tree votes on whether an order will match quickly or not.
            - We average these votes to get a final probability.
            - Because we used synthetic subsets in each iteration, you saw different test accuracies as we progressed.
            """
            )
    else:
        st.info("Press 'Generate & Train Model' to see data and training progress.")


# ------------------------------------------------------------------------
# 3) PREDICT & PRIORITIZE
# ------------------------------------------------------------------------
with tabs[2]:
    st.header("🔮 Predict Match Probability & Prioritize")
    st.sidebar.header("New Order Controls")
    n_new = st.sidebar.slider("Number of new orders", 50, 2000, 200)
    seed_infer = st.sidebar.number_input("Random Seed (Inference)", value=123, step=1)

    if (
        "trained_model" not in st.session_state
        or st.session_state["trained_model"] is None
    ):
        st.error("No trained model found. Go to 'Train AI Model' tab and run training.")
        st.stop()

    # Generate new orders
    df_new = generate_new_orders(n_new, seed_infer)
    st.markdown("### New Unlabeled Orders (Sample)")
    st.dataframe(df_new.head(10))

    model_final, scaler = st.session_state["trained_model"]

    # One-hot encode side/ticker
    df_enc = df_new.copy()
    df_enc = pd.get_dummies(df_enc, columns=["side", "ticker"])

    # We assume we had columns like side_BUY, side_SELL, ticker_AAPL, ticker_TSLA, etc.
    # Let's see what columns the model might expect:
    if hasattr(scaler, "feature_names_in_"):
        needed_cols = list(scaler.feature_names_in_)
    else:
        # If not stored, guess from training:
        # We'll retrieve from the "training_data" if stored
        if (
            "training_data" in st.session_state
            and st.session_state["training_data"] is not None
        ):
            train_df = st.session_state["training_data"]
            # we have columns ["side_BUY", "side_SELL", "ticker_AAPL", ...]
            # Let's do a quick dummies approach for that as well and see:
            train_df_enc = pd.get_dummies(train_df, columns=["side", "ticker"])
            needed_cols = train_df_enc.drop(
                ["order_id", "fast_match"], axis=1
            ).columns.tolist()
        else:
            needed_cols = df_enc.columns.tolist()

    # We remove columns not needed
    for c in ["order_id", "timestamp", "latency_ms", "price", "size"]:
        # Actually we do want price/size in the model. Let's keep them if the model was trained on them
        # We'll just unify the approach: only remove "order_id", "timestamp" if they're not in needed_cols
        if c not in needed_cols and c in df_enc.columns:
            df_enc.drop(c, axis=1, inplace=True)

    # Add missing columns as 0
    for col in needed_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    # Reindex
    df_enc = df_enc.reindex(columns=needed_cols, fill_value=0)

    X_infer = scaler.transform(df_enc)
    probas = model_final.predict_proba(X_infer)
    df_new["match_probability"] = probas[:, 1]

    # Sort descending by match_probability
    df_ai = df_new.sort_values("match_probability", ascending=False)
    df_fifo = df_new.sort_values("timestamp")

    st.markdown("### AI-Prioritized Orders (Highest Probability First)")
    st.dataframe(df_ai.head(10))

    # 2D chart: price vs. size, color=prob, size=prob
    st.subheader("2D Visualization: Price vs. Size (Colored by Predicted Probability)")
    fig_2d = px.scatter(
        df_new,
        x="price",
        y="size",
        color="match_probability",
        size="match_probability",
        hover_data=["side", "ticker", "latency_ms"],
        title="Scatter: Price vs. Size, with Probability as color & size",
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    st.session_state["df_fifo"] = df_fifo
    st.session_state["df_ai"] = df_ai
    st.success("Orders are ready for final trading simulation.")


# ------------------------------------------------------------------------
# 4) RUN TRADING SIMULATION
# ------------------------------------------------------------------------
with tabs[3]:
    st.header("🚀 Trading Simulation")

    st.markdown(
        """
    Now we'll insert these **new orders** into a limit order book in two ways:
    1. **FIFO**: By arrival time.
    2. **AI**: By descending match_probability.
    
    Then we measure how many fill events occur (toy metric).
    """
    )

    partial_fill = st.checkbox("Allow Partial Fills?", True)

    if "df_fifo" not in st.session_state or "df_ai" not in st.session_state:
        st.error("No new orders to simulate. Go to 'Predict & Prioritize' tab first.")
        st.stop()

    df_fifo = st.session_state["df_fifo"].copy()
    df_ai = st.session_state["df_ai"].copy()

    if st.button("Run Trading Simulation Now!"):
        fill_counts_fifo = []
        all_fills_fifo = pd.DataFrame()
        cumulative_orders_fifo = []

        # FIFO approach
        for i, row in enumerate(df_fifo.itertuples()):
            time.sleep(0.000005)  # micro-latency
            cumulative_orders_fifo.append(
                {
                    "order_id": row.order_id,
                    "side": row.side,
                    "price": row.price,
                    "size": row.size,
                }
            )
            df_co = pd.DataFrame(cumulative_orders_fifo)
            fill_df = basic_order_book_engine(df_co, partial_fill=partial_fill)
            fill_counts_fifo.append(len(fill_df))
            if i == len(df_fifo) - 1:
                all_fills_fifo = fill_df

        # AI approach
        fill_counts_ai = []
        all_fills_ai = pd.DataFrame()
        cumulative_orders_ai = []
        for i, row in enumerate(df_ai.itertuples()):
            time.sleep(0.000005)
            cumulative_orders_ai.append(
                {
                    "order_id": row.order_id,
                    "side": row.side,
                    "price": row.price,
                    "size": row.size,
                }
            )
            df_co_ai = pd.DataFrame(cumulative_orders_ai)
            fill_df_ai = basic_order_book_engine(df_co_ai, partial_fill=partial_fill)
            fill_counts_ai.append(len(fill_df_ai))
            if i == len(df_ai) - 1:
                all_fills_ai = fill_df_ai

        # Compare
        df_compare = pd.DataFrame(
            {
                "num_orders_processed": range(1, len(df_fifo) + 1),
                "fills_fifo": fill_counts_fifo,
                "fills_ai": fill_counts_ai,
            }
        )

        st.subheader("Cumulative Fill Events vs. Orders Processed")
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=df_compare["num_orders_processed"],
                y=df_compare["fills_fifo"],
                mode="lines+markers",
                name="FIFO",
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=df_compare["num_orders_processed"],
                y=df_compare["fills_ai"],
                mode="lines+markers",
                name="AI Priority",
            )
        )
        fig_line.update_layout(
            title="Cumulative Fill Events vs. Number of Orders Processed",
            xaxis_title="Orders Processed",
            yaxis_title="Total Fills",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        final_fifo_fills = df_compare["fills_fifo"].iloc[-1]
        final_ai_fills = df_compare["fills_ai"].iloc[-1]
        st.write(
            f"**Final Fill Count**: FIFO = {final_fifo_fills}, AI = {final_ai_fills}"
        )

        with st.expander("Show FIFO Fill Events"):
            st.dataframe(all_fills_fifo.head(50))
        with st.expander("Show AI Fill Events"):
            st.dataframe(all_fills_ai.head(50))

    else:
        st.info("Press the button above to run the toy simulation.")


# ------------------------------------------------------------------------
# 5) MEETING WITH STEVE
# ------------------------------------------------------------------------
with tabs[4]:
    st.header("👓 Meeting with Steve Jobs (Design Perspective)")

    st.markdown(
        """
    After a quick meeting with "Steve Jobs", here are some **design & communication** pointers 
    for explaining this AI concept to **non-data science** or **non-technical** audiences:

    1. **Simplicity**: Show a **single** chart or storyline at a time. 
       The user sees each step in plain language – *"We have 2 approaches: FIFO vs. AI. 
       Watch how AI approach leads to faster partial fills."*
    2. **Intuitive Visuals**: 
       - A bright color for flagged “high probability” orders. 
       - Possibly an animation or stepwise highlight in a limit order book diagram 
         to illustrate how certain orders cross faster.
    3. **Use Familiar Language**: 
       - Instead of “RandomForestClassifier,” say “We used a machine pattern system 
         that learned from past trades to guess which new trades will fill quickly.”
    4. **Focus on *Why** it Matters**: 
       - For a non-technical person, emphasize *business impact*: 
         “This AI can reduce trade latency by 20-40%, 
         which can mean millions in HFT edge for some firms.”
    5. **Iterative / Interactive**: 
       - Provide sliders (like we do) for # of orders, noise, partial fill setting, etc. 
         The user can *see* how changes affect the fill chart in real time.
    6. **Limit Complexity**: 
       - Summarize metrics in a few big numbers or short bullet points. 
       - Hide or collapse advanced details behind “More Info” expanders if needed.

    By applying these suggestions, we **bridge** the gap between **AI intricacies** and 
    the **practical benefits** that matter to business stakeholders, HFT participants, 
    and everyday users alike.
    """
    )

    st.success("End of Demo. Adjust parameters & enjoy experimenting!")
