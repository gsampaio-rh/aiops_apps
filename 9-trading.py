import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# For anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -------------- Page Setup --------------
st.set_page_config(page_title="Anomaly Detection Tutorial", layout="wide")

st.title("💹 Advanced AI for Anomaly Detection in Trading")

# ------- Define Tabs -------
tabs = st.tabs(
    [
        "🏠 Home",
        "📈 Trade Data",
        "🚨 Anomaly Detection & Explanation",
        "📊 Detailed Breakdown",
    ]
)

# =============== Trader Names ===============
TRADER_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Gina",
    "Hank",
    "Ivy",
    "James",
    "Karen",
    "Leo",
    "Mona",
    "Nick",
    "Olivia",
    "Paul",
    "Queen",
    "Ray",
    "Sara",
    "Tom",
    "Uma",
    "Victor",
    "Wendy",
    "Xander",
    "Yara",
    "Zack",
]


def get_trader_name(trader_id, seed_val=42):
    """
    Deterministically pick a human-friendly name for each trader_id
    so the same ID always maps to the same name.
    If we run out of names, we append a number.
    """
    random.seed(trader_id + seed_val)
    idx = trader_id % len(TRADER_NAMES)
    base_name = TRADER_NAMES[idx]
    suffix_num = trader_id // len(TRADER_NAMES)
    if suffix_num > 0:
        base_name += f"_{suffix_num}"
    return base_name


# =============== Synthetic Data Generation ===============
@st.cache_data
def generate_synthetic_trades(n_traders, anomaly_prob, seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    data = []
    trade_id_counter = 1

    for trader_id in range(n_traders):
        suspicious = np.random.rand() < anomaly_prob
        # Suspicious traders have more trades with bigger volumes
        n_trades = (
            np.random.randint(20, 40) if not suspicious else np.random.randint(50, 80)
        )

        for _ in range(n_trades):
            if suspicious:
                volume = np.random.normal(5000, 1500)  # bigger trades
                success_chance = 0.80  # suspicious might fail more often
            else:
                volume = np.random.normal(1000, 300)
                success_chance = 0.90

            timestamp = datetime.now() - pd.Timedelta(np.random.randint(1, 72), "h")
            counterparty = np.random.randint(0, n_traders)
            while counterparty == trader_id:
                counterparty = np.random.randint(0, n_traders)

            succeeded = np.random.rand() < success_chance

            data.append(
                {
                    "trade_id": trade_id_counter,
                    "trader_id": trader_id,
                    "trader_name": get_trader_name(trader_id, seed_val),
                    "counterparty_id": counterparty,
                    "volume": float(max(10, abs(volume))),
                    "timestamp": timestamp,
                    "true_suspicious_trader": suspicious,
                    "trade_succeeded": succeeded,
                }
            )
            trade_id_counter += 1
    df = pd.DataFrame(data)
    return df


# =============== HOME TAB ===============
with tabs[0]:
    st.header("Welcome to the Tutorial")
    st.markdown(
        """
        This demo simulates trades and uses **multiple features** plus an 
        **Isolation Forest** for anomaly detection. We aim to see *why* each trader is flagged.
        
        **Tutorial Steps**:
        1. **Trade Data**: View the synthetic trades and basic stats.
        2. **Anomaly Detection**: See how we compute advanced features 
           (mean trade size, success ratio, time-between-trades) and run an Isolation Forest.
        3. **Detailed Breakdown**: Explore expansions and time-series patterns for flagged traders.
        """
    )

    st.sidebar.header("Simulation Settings")
    num_traders = st.sidebar.slider("Number of Traders", 10, 200, 50)
    anomaly_probability = st.sidebar.slider(
        "Probability a Trader is 'Suspicious'", 0.0, 1.0, 0.20
    )
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    # Generate or update trades in session state
    if "trades" not in st.session_state:
        st.session_state["trades"] = generate_synthetic_trades(
            num_traders, anomaly_probability, seed
        )
    else:
        st.session_state["trades"] = generate_synthetic_trades(
            num_traders, anomaly_probability, seed
        )

# =============== TRADE DATA TAB ===============
with tabs[1]:
    st.header("📈 Synthetic Trade Data")
    df = st.session_state["trades"].copy()

    st.markdown("**Preview of the trade dataset:**")
    st.dataframe(df, use_container_width=True)

    total_trades = len(df)
    unique_traders = df["trader_id"].nunique()
    suspicious_traders = df["true_suspicious_trader"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", total_trades)
    col2.metric("Unique Traders", unique_traders)
    col3.metric("Suspicious Traders", suspicious_traders)

    st.subheader("Trade Volume Distribution (Average per Trader)")
    # Group by trader, get average volume
    group_vol = df.groupby("trader_id")["volume"].mean().reset_index()
    group_vol["susp"] = df.groupby("trader_id")["true_suspicious_trader"].max().values

    fig_hist = px.histogram(
        group_vol,
        x="volume",
        color="susp",
        nbins=30,
        labels={"volume": "Avg Trade Volume", "susp": "Suspicious?"},
        title="Histogram of Average Trade Volume per Trader",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =============== ANOMALY DETECTION & EXPLANATION TAB ===============
with tabs[2]:
    st.header("🚨 Anomaly Detection & Explanation")

    df = st.session_state["trades"].copy()

    st.markdown(
        """
        **Multiple Features** used for each trader:
        - **Total Volume** (sum of volumes)
        - **Trade Count** (number of trades)
        - **Mean Trade Size** = total_volume / trade_count
        - **Success Ratio** = (# successes) / (trade_count)
        - **Time Between Trades** (average seconds between trades)
        
        We'll feed these features into an **Isolation Forest** to detect anomalies.
        """
    )

    # ------------------- 1) Aggregate Trader-Level Features --------------------
    grouped = (
        df.groupby(["trader_id", "trader_name"])
        .agg(
            total_volume=("volume", "sum"),
            trade_count=("trade_id", "count"),
            num_success=("trade_succeeded", "sum"),
            first_trade=("timestamp", "min"),
            last_trade=("timestamp", "max"),
            true_susp=("true_suspicious_trader", "max"),
        )
        .reset_index()
    )

    grouped["mean_trade_size"] = grouped["total_volume"] / grouped["trade_count"]
    grouped["success_ratio"] = grouped["num_success"] / grouped["trade_count"]
    grouped["time_between_trades"] = (
        (grouped["last_trade"] - grouped["first_trade"]).dt.total_seconds()
        / grouped["trade_count"]
    ).fillna(0)

    # ------------------- 2) Isolation Forest --------------------
    st.subheader("Isolation Forest Model")

    features = [
        "total_volume",
        "trade_count",
        "mean_trade_size",
        "success_ratio",
        "time_between_trades",
    ]
    X = grouped[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit isolation forest
    model = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=seed  # ~10% anomalies
    )
    model.fit(X_scaled)

    # Predictions: -1 => anomaly, 1 => normal
    y_pred = model.predict(X_scaled)
    grouped["flagged"] = y_pred == -1

    # For a "suspicious_score" style metric, we can use -1 * decision_function
    # so higher = more likely anomaly
    anomaly_scores = model.decision_function(X_scaled)  # higher => more normal
    grouped["suspicious_score"] = -anomaly_scores  # so higher => more suspicious

    num_flagged = grouped["flagged"].sum()
    st.write(
        f"**Isolation Forest** flagged {num_flagged} out of {len(grouped)} traders as anomalies."
    )

    # Display the new "grouped" table with the computed features
    st.markdown("**Trader-Level Features & Flags**")
    st.dataframe(
        grouped[
            [
                "trader_id",
                "trader_name",
                "total_volume",
                "trade_count",
                "mean_trade_size",
                "success_ratio",
                "time_between_trades",
                "flagged",
                "true_susp",
                "suspicious_score",
            ]
        ].sort_values("suspicious_score", ascending=False),
        use_container_width=True,
    )

    # ------------------- 3) Distribution of Suspicious Score --------------------
    st.subheader("Distribution of Suspicious Scores")
    st.markdown(
        "Higher = more suspicious (since we took the negative of the model’s decision_function)."
    )
    fig_hist2 = px.histogram(
        grouped,
        x="suspicious_score",
        nbins=30,
        color="flagged",
        labels={"flagged": "Flagged?"},
        title="Histogram of Suspicious Scores (Isolation Forest)",
    )
    st.plotly_chart(fig_hist2, use_container_width=True)

    scores = grouped["suspicious_score"]

    # Option 1: shift everything by the min plus a small offset
    min_val = scores.min()  # could be negative
    marker_sizes = (scores - min_val) + 1

    # ------------------- 4) Scatter Plot of Two Chosen Features --------------------
    # Let's pick "total_volume" vs "trade_count" for a quick 2D view
    st.subheader("Scatter: Total Volume vs. Trade Count (Isolation Forest)")
    fig_scatter = px.scatter(
        grouped,
        x="total_volume",
        y="trade_count",
        color="flagged",
        hover_data=["trader_name", "suspicious_score"],
        title="Volume vs. Trade Count (Flagged in color)",
        labels={
            "flagged": "Flagged?",
            "total_volume": "Total Volume",
            "trade_count": "Trade Count",
        },
        size=marker_sizes,  # your new, always-positive series
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------- 5) Confusion Matrix / Ground-Truth Check --------------------
    st.subheader("Confusion Matrix (Toy Ground Truth Check)")
    grouped["TP"] = grouped["true_susp"] & grouped["flagged"]
    grouped["FP"] = (~grouped["true_susp"]) & grouped["flagged"]
    grouped["FN"] = grouped["true_susp"] & (~grouped["flagged"])
    grouped["TN"] = (~grouped["true_susp"]) & (~grouped["flagged"])

    tp = grouped["TP"].sum()
    fp = grouped["FP"].sum()
    fn = grouped["FN"].sum()
    tn = grouped["TN"].sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    colA, colB, colC, colD = st.columns(4)
    colA.metric("True Positives", tp)
    colB.metric("False Positives", fp)
    colC.metric("Precision", f"{precision:.2f}")
    colD.metric("Recall", f"{recall:.2f}")

    st.info(
        "You can tweak the `contamination` parameter in Isolation Forest or examine `decision_function` scores to adjust how many outliers are flagged."
    )

    # Store the updated grouped in session for the next tab
    st.session_state["grouped"] = grouped
    st.session_state["df"] = df  # original trades

# =============== DETAILED BREAKDOWN TAB ===============
with tabs[3]:
    st.markdown("## Detailed Breakdown of Flagged Traders")

    if "grouped" not in st.session_state:
        st.warning("Please run the Anomaly Detection tab first.")
        st.stop()

    grouped = st.session_state["grouped"]
    df = st.session_state["df"]

    flagged_df = grouped[grouped["flagged"] == True].copy()
    top_n = st.slider("Show Top N Anomalies", 1, max(1, len(flagged_df)), 5)
    flagged_sorted = flagged_df.sort_values("suspicious_score", ascending=False).head(
        top_n
    )

    st.markdown("### Expand for Each Flagged Trader's Details")
    for idx, row in flagged_sorted.iterrows():
        t_id = row["trader_id"]
        t_name = row["trader_name"]
        tv = row["total_volume"]
        tc = row["trade_count"]
        mts = row["mean_trade_size"]
        sr = row["success_ratio"]
        tbt = row["time_between_trades"]
        score = row["suspicious_score"]
        ground_truth = row["true_susp"]

        with st.expander(f"Trader {t_name} (ID={t_id}) — Score = {score:.2f}"):
            st.write(f"**Total Volume** = {tv:,.2f},  **Trade Count** = {tc}")
            st.write(f"**Mean Trade Size** = {mts:,.2f}")
            st.write(f"**Success Ratio** = {sr:.2f}")
            st.write(f"**Time Between Trades** = {tbt:.2f} sec/trade")
            st.write(f"**Suspicious Score** = {score:.2f}")
            st.warning(f"**Flagged** by Isolation Forest => (score ~ {score:.2f}).")
            if ground_truth:
                st.success(
                    "Ground Truth: This trader **is** suspicious. (True Positive)"
                )
            else:
                st.error(
                    "Ground Truth: This trader is **not** suspicious. (False Positive?)"
                )

    st.subheader("Compare One Suspect to Mean Lines (Volume vs. Count)")
    flagged_traders = flagged_sorted["trader_id"].tolist()
    if not flagged_traders:
        st.info("No traders flagged as anomalies under the current settings.")
    else:
        selected_suspect = st.selectbox(
            "Select a flagged trader to highlight",
            flagged_traders,
            format_func=lambda tid: grouped.loc[
                grouped["trader_id"] == tid, "trader_name"
            ].values[0],
        )

        mean_vol = grouped["total_volume"].mean()
        mean_count = grouped["trade_count"].mean()

        # Mark color on each trader
        def color_picker(row):
            if row["trader_id"] == selected_suspect:
                return "SELECTED_SUSPECT"
            elif row["flagged"]:
                return "FLAGGED_OTHERS"
            else:
                return "NORMAL"

        grouped["color_label"] = grouped.apply(color_picker, axis=1)

        fig_compare = px.scatter(
            grouped,
            x="total_volume",
            y="trade_count",
            color="color_label",
            hover_data=["trader_name", "suspicious_score"],
            title="Suspect vs. Mean Comparison (Volume vs. Count)",
            color_discrete_map={
                "SELECTED_SUSPECT": "red",
                "FLAGGED_OTHERS": "orange",
                "NORMAL": "blue",
            },
            labels={"total_volume": "Total Volume", "trade_count": "Trade Count"},
        )

        # Add lines for global means
        fig_compare.add_vline(
            x=mean_vol,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean Volume={mean_vol:.2f}",
        )
        fig_compare.add_hline(
            y=mean_count,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean Count={mean_count:.2f}",
        )

        # Enlarge marker for the selected suspect
        new_marker_sizes = []
        for _, r in grouped.iterrows():
            if r["trader_id"] == selected_suspect:
                new_marker_sizes.append(14)
            else:
                new_marker_sizes.append(8)
        fig_compare.update_traces(
            marker=dict(size=new_marker_sizes), selector=dict(mode="markers")
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # Show numeric deviation
        suspect_row = grouped[grouped["trader_id"] == selected_suspect].iloc[0]
        st.write(
            f"**Trader {suspect_row['trader_name']}**: "
            f"Volume = {suspect_row['total_volume']:.2f}, "
            f"Trade Count = {suspect_row['trade_count']:.2f}"
        )
        st.write(
            f"**Deviation from mean**: "
            f"Volume is {(suspect_row['total_volume'] - mean_vol):.2f} away from {mean_vol:.2f}, "
            f"Count is {(suspect_row['trade_count'] - mean_count):.2f} away from {mean_count:.2f}."
        )

    st.subheader("Suspect's Trade Patterns Over Time")
    flagged_traders_all = (
        grouped[grouped["flagged"] == True]["trader_id"].unique().tolist()
    )
    if len(flagged_traders_all) == 0:
        st.info("No traders flagged as anomalies under the current threshold.")
    else:
        selected_suspect2 = st.selectbox(
            "Select a flagged trader to see time-series comparison:",
            flagged_traders_all,
            format_func=lambda tid: grouped.loc[
                grouped["trader_id"] == tid, "trader_name"
            ].values[0],
            key="time_series_selectbox",  # unique key if reusing variable
        )

        # Prepare time bin for trades
        df["hour_bin"] = df["timestamp"].dt.floor("H")
        suspect_df = df[df["trader_id"] == selected_suspect2].copy()

        # Global aggregates (all traders)
        global_vol_per_hour = (
            df.groupby("hour_bin")["volume"]
            .mean()
            .reset_index(name="global_avg_volume")
        )
        global_count_per_hour = (
            df.groupby("hour_bin")["trade_id"]
            .count()
            .reset_index(name="global_trade_count")
        )

        # Suspect aggregates
        suspect_vol_per_hour = (
            suspect_df.groupby("hour_bin")["volume"]
            .mean()
            .reset_index(name="suspect_avg_volume")
        )
        suspect_count_per_hour = (
            suspect_df.groupby("hour_bin")["trade_id"]
            .count()
            .reset_index(name="suspect_trade_count")
        )

        # Merge so we keep all hour bins
        merged_volume = pd.merge(
            global_vol_per_hour, suspect_vol_per_hour, on="hour_bin", how="outer"
        )
        merged_count = pd.merge(
            global_count_per_hour, suspect_count_per_hour, on="hour_bin", how="outer"
        )

        merged_volume.fillna(0, inplace=True)
        merged_count.fillna(0, inplace=True)

        # Plot volume over time
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=merged_volume["hour_bin"],
                y=merged_volume["global_avg_volume"],
                mode="lines+markers",
                name="Global Avg Volume",
            )
        )
        fig_vol.add_trace(
            go.Scatter(
                x=merged_volume["hour_bin"],
                y=merged_volume["suspect_avg_volume"],
                mode="lines+markers",
                name=f"Suspect {selected_suspect2} Avg Volume",
            )
        )
        fig_vol.update_layout(
            title=f"Volume Over Time (Hourly Bin) - Trader {selected_suspect2}",
            xaxis_title="Time (Hour)",
            yaxis_title="Average Volume",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Plot trade count over time
        fig_cnt = go.Figure()
        fig_cnt.add_trace(
            go.Scatter(
                x=merged_count["hour_bin"],
                y=merged_count["global_trade_count"],
                mode="lines+markers",
                name="Global Trade Count",
            )
        )
        fig_cnt.add_trace(
            go.Scatter(
                x=merged_count["hour_bin"],
                y=merged_count["suspect_trade_count"],
                mode="lines+markers",
                name=f"Suspect {selected_suspect2} Trade Count",
            )
        )
        fig_cnt.update_layout(
            title=f"Trade Count Over Time (Hourly Bin) - Trader {selected_suspect2}",
            xaxis_title="Time (Hour)",
            yaxis_title="Number of Trades",
        )
        st.plotly_chart(fig_cnt, use_container_width=True)

        st.markdown(
            f"""
            **Interpretation**: If Trader {selected_suspect2} consistently shows higher 
            (or spikier) volume/frequency than the global average at the same times, 
            it may confirm why they're flagged.
            """
        )
