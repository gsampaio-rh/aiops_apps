import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------- Page Setup --------------
st.set_page_config(page_title="Anomaly Detection Tutorial", layout="wide")

st.title("💹 AI for Anomaly Detection in Trading")

tabs = st.tabs(
    [
        "🏠 Home",
        "📈 Trade Data",
        "🚨 Anomaly Detection & Explanation",
        "📊 Detailed Breakdown",
    ]
)

# =======================================
# 1) HOME TAB
# =======================================
with tabs[0]:
    st.header("Welcome to the Tutorial")
    st.markdown(
        """
        This demo simulates trades and then **flags anomalous traders** using 
        a simple formula. The goal is to see *why* each trader is flagged.
        
        **Tutorial Steps**:
        1. **Trade Data**: View the synthetic trades and basic stats.
        2. **Anomaly Detection**: See how we compute each trader's “suspicious_score” 
           and interpret why they're flagged.
        """
    )

    st.sidebar.header("Simulation Settings")
    num_traders = st.sidebar.slider("Number of Traders", 10, 200, 50)
    anomaly_probability = st.sidebar.slider(
        "Probability a Trader is 'Suspicious'",
        0.0, 1.0, 0.20
    )
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    @st.cache_data
    def generate_synthetic_trades(n_traders, anomaly_prob, seed_val):
        np.random.seed(seed_val)
        data = []
        trade_id_counter = 1
        for trader_id in range(n_traders):
            suspicious = np.random.rand() < anomaly_prob
            # suspicious traders appear in more trades with bigger volumes
            n_trades = np.random.randint(20, 40) if not suspicious else np.random.randint(50, 80)

            for _ in range(n_trades):
                if suspicious:
                    volume = np.random.normal(4000, 1000)  # bigger trades
                else:
                    volume = np.random.normal(600, 200)

                # some random timestamp
                timestamp = datetime.now() - pd.Timedelta(np.random.randint(1, 72), "h")

                # random counterparty
                c_id = np.random.randint(0, n_traders)
                while c_id == trader_id:
                    c_id = np.random.randint(0, n_traders)

                data.append({
                    "trade_id": trade_id_counter,
                    "trader_id": trader_id,
                    "counterparty_id": c_id,
                    "volume": float(max(10, abs(volume))),
                    "timestamp": timestamp,
                    "true_suspicious_trader": suspicious
                })
                trade_id_counter += 1
        return pd.DataFrame(data)

    # Generate or update trades in session state
    if "trades" not in st.session_state:
        st.session_state["trades"] = generate_synthetic_trades(num_traders, anomaly_probability, seed)
    else:
        st.session_state["trades"] = generate_synthetic_trades(num_traders, anomaly_probability, seed)

# =======================================
# 2) TRADE DATA TAB
# =======================================
with tabs[1]:
    st.header("📈 Synthetic Trade Data")
    df = st.session_state["trades"]
    st.markdown("**Preview of the trade dataset:**")
    st.dataframe(df.head(30))

    total_trades = len(df)
    unique_traders = df["trader_id"].nunique()
    suspicious_traders = df["true_suspicious_trader"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", total_trades)
    col2.metric("Unique Traders", unique_traders)
    col3.metric("Suspicious Traders", suspicious_traders)

    st.subheader("Trade Volume Distribution")
    # Group by trader, get average volume
    group_vol = df.groupby("trader_id")["volume"].mean().reset_index()
    group_vol["susp"] = df.groupby("trader_id")["true_suspicious_trader"].max().values

    fig_hist = px.histogram(
        group_vol,
        x="volume",
        color="susp",
        nbins=30,
        labels={"volume": "Avg Trade Volume", "susp": "Suspicious"},
        title="Histogram of Average Trade Volume per Trader"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =======================================
# 3) ANOMALY DETECTION & EXPLANATION
# =======================================
with tabs[2]:
    st.header("🚨 Anomaly Detection & Explanation")

    df = st.session_state["trades"].copy()

    st.markdown(
        """
        In this demo, each trader has two features:
        - **Total Volume**: The sum of all trade volumes for that trader.
        - **Trade Count**: How many trades they've participated in.
        """
    )

    # --------------------------------------------------
    # 1. Compute Features for Each Trader
    # --------------------------------------------------
    grouped = (
        df.groupby("trader_id")
        .agg(
            total_volume=("volume", "sum"),
            trade_count=("trade_id", "count"),
            true_susp=("true_suspicious_trader", "max"),
        )
        .reset_index()
    )

    # Calculate means & standard deviations
    mean_vol = grouped["total_volume"].mean()
    std_vol = grouped["total_volume"].std(ddof=1)
    mean_count = grouped["trade_count"].mean()
    std_count = grouped["trade_count"].std(ddof=1)

    # Z-scores
    grouped["z_volume"] = (grouped["total_volume"] - mean_vol) / std_vol
    grouped["z_count"] = (grouped["trade_count"] - mean_count) / std_count
    grouped["suspicious_score"] = grouped["z_volume"] + grouped["z_count"]

    # --------------------------------------------------
    # 2. Show Mean & Std Dev
    # --------------------------------------------------
    st.subheader("Overall Stats")
    colM1, colM2 = st.columns(2)
    with colM1:
        st.metric("Mean of Total Volume", f"{mean_vol:.2f}")
        st.metric("Std Dev of Total Volume", f"{std_vol:.2f}")
    with colM2:
        st.metric("Mean of Trade Count", f"{mean_count:.2f}")
        st.metric("Std Dev of Trade Count", f"{std_count:.2f}")

    st.write(
        """
    The **z-scores** measure how far each trader is from these means, 
    scaled by the standard deviations.
    """
    )

    # --------------------------------------------------
    # 3. Let user pick threshold & flag traders
    # --------------------------------------------------
    default_threshold = float(
        grouped["suspicious_score"].quantile(0.9)
    )  # 90th percentile
    threshold = st.slider(
        "Suspicious Score Threshold",
        float(grouped["suspicious_score"].min()),
        float(grouped["suspicious_score"].max()),
        default_threshold,
        help="Traders above this threshold are flagged.",
    )
    grouped["flagged"] = grouped["suspicious_score"] > threshold

    # --------------------------------------------------
    # 4. Show Full Computation Table
    # --------------------------------------------------
    st.subheader("Full Computation Table")
    st.markdown(
        """
    Below you can see each trader's computed values:
    """
    )
    st.dataframe(
        grouped[
            [
                "trader_id",
                "total_volume",
                "trade_count",
                "z_volume",
                "z_count",
                "suspicious_score",
                "flagged",
                "true_susp",
            ]
        ].sort_values("suspicious_score", ascending=False),
        use_container_width=True,
    )

    # --------------------------------------------------
    # 5. Distribution of suspicious_score
    # --------------------------------------------------
    st.subheader("Distribution of Suspicious Scores")
    st.markdown(
        "Traders on the far right are more deviant. The vertical line shows your chosen threshold."
    )

    fig_hist = px.histogram(
        grouped,
        x="suspicious_score",
        nbins=30,
        color="flagged",
        title="Histogram of Suspicious Scores (color = flagged)",
        labels={"flagged": "Flagged?"},
    )
    fig_hist.add_vline(
        x=threshold,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold={threshold:.2f}",
        annotation_position="top left",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --------------------------------------------------
    # 6. Scatter Plot: z_volume vs. z_count
    # --------------------------------------------------
    st.subheader("Z-Score Scatter Plot (Volume vs. Trade Count)")
    st.markdown(
        "Points in the upper-right corner deviate more from the mean in both volume and count."
    )

    fig_scatter = px.scatter(
        grouped,
        x="z_volume",
        y="z_count",
        color="flagged",
        hover_data=["trader_id", "suspicious_score", "total_volume", "trade_count"],
        labels={"z_volume": "Z-Score (Volume)", "z_count": "Z-Score (Count)"},
        title="z_volume vs. z_count (Colored by Flagged)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------------------------------------
    # 9) Basic Confusion Matrix
    # --------------------------------------------------
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
        "Adjust the threshold to see how it affects false positives & false negatives."
    )

with tabs[3]:
    # --------------------------------------------------
    # 8) Expanders: Detailed Math for Each Flagged Trader
    # --------------------------------------------------
    st.markdown("### Detailed Breakdown of Flagged Traders")
    flagged_df = grouped[grouped["flagged"] == True].copy()
    top_n = st.slider("Show Top N Anomalies", 1, max(1, len(flagged_df)), 5)
    flagged_sorted = flagged_df.sort_values("suspicious_score", ascending=False).head(
        top_n
    )

    for idx, row in flagged_sorted.iterrows():
        trader_id = row["trader_id"]
        tv = row["total_volume"]
        tc = row["trade_count"]
        zv = row["z_volume"]
        zc = row["z_count"]
        score = row["suspicious_score"]
        ground_truth = row["true_susp"]

        with st.expander(f"Trader {trader_id} — Score = {score:.2f}"):
            st.write(f"**Total Volume** = {tv:,.2f},  **Trade Count** = {tc}")
            st.write(f"**z_volume** = {zv:.2f}, **z_count** = {zc:.2f}")
            st.write(
                f"**score** = z_volume + z_count = {zv:.2f} + {zc:.2f} = {score:.2f}"
            )
            st.warning(f"**Flagged** because {score:.2f} > {threshold:.2f}.")
            if ground_truth:
                st.success(
                    "Ground Truth: This trader **is** suspicious. (True Positive)"
                )
            else:
                st.error(
                    "Ground Truth: This trader **is not** suspicious. (False Positive?)"
                )

    st.subheader("Compare One Suspect to Mean Lines")
    # --- 1) Let user pick which flagged suspect to highlight ---
    flagged_traders = grouped[grouped["flagged"] == True]["trader_id"].tolist()
    if not flagged_traders:
        st.info("No traders flagged as anomalies under the current threshold.")
    else:
        selected_suspect = st.selectbox(
            "Select a flagged trader to highlight", flagged_traders
        )

        # --- 2) Build the scatter chart for all traders ---
        # Create a color column that marks the selected suspect
        def color_picker(row):
            if row["trader_id"] == selected_suspect:
                return "SELECTED_SUSPECT"
            elif row["flagged"]:
                return "FLAGGED_OTHERS"
            else:
                return "NORMAL"

        grouped["color_label"] = grouped.apply(color_picker, axis=1)

        # Plotly Express scatter
        fig_compare = px.scatter(
            grouped,
            x="total_volume",
            y="trade_count",
            color="color_label",
            hover_data=["trader_id", "flagged"],
            title="Suspect vs. Mean Comparison",
            color_discrete_map={
                "SELECTED_SUSPECT": "red",
                "FLAGGED_OTHERS": "orange",
                "NORMAL": "blue",
            },
            labels={"total_volume": "Total Volume", "trade_count": "Trade Count"},
        )

        # --- 3) Add reference lines for means ---
        # Vertical line at mean volume
        fig_compare.add_vline(
            x=mean_vol,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean Volume={mean_vol:.2f}",
            annotation_position="top right",
        )
        # Horizontal line at mean count
        fig_compare.add_hline(
            y=mean_count,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean Count={mean_count:.2f}",
            annotation_position="bottom left",
        )

        # --- 4) Emphasize the selected suspect with a bigger marker size ---
        # We'll update marker sizes based on color_label
        new_marker_sizes = []
        for idx, row in grouped.iterrows():
            if row["trader_id"] == selected_suspect:
                new_marker_sizes.append(14)  # bigger for the suspect
            else:
                new_marker_sizes.append(8)  # normal size

        # Update the figure's marker sizes
        fig_compare.update_traces(
            marker=dict(size=new_marker_sizes), selector=dict(mode="markers")
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # --- 5) Explanation or numeric details ---
        # Pull out the suspect's row
        suspect_row = grouped[grouped["trader_id"] == selected_suspect].iloc[0]
        st.write(
            f"**Trader {selected_suspect}** has Total Volume = {suspect_row['total_volume']:.2f}, "
            f"Trade Count = {suspect_row['trade_count']:.2f}"
        )
        st.write(
            f"**Deviation from mean**: Volume is {(suspect_row['total_volume'] - mean_vol):.2f} away from {mean_vol:.2f}, "
            f"Count is {(suspect_row['trade_count'] - mean_count):.2f} away from {mean_count:.2f}."
        )

    st.subheader("Suspect's Trade Patterns Over Time")

    # 1) Check if we have any flagged traders:
    flagged_traders = grouped[grouped["flagged"] == True]["trader_id"].unique().tolist()
    if len(flagged_traders) == 0:
        st.info("No traders flagged as anomalies under the current threshold.")
    else:
        selected_suspect = st.selectbox(
            "Select a flagged trader to see time-series comparison:", flagged_traders
        )

        # 2) Filter the suspect's trades & create a time bin (e.g., hourly)
        df["hour_bin"] = df["timestamp"].dt.floor("H")  # or "D" for daily
        suspect_df = df[df["trader_id"] == selected_suspect].copy()

        # 3) Compute global aggregates by time
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

        # 4) Compute suspect aggregates by time
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

        # 5) Merge data so we can plot it easily
        #    We'll do an outer join so we don't lose hours that exist in one but not the other
        merged_volume = pd.merge(
            global_vol_per_hour, suspect_vol_per_hour, on="hour_bin", how="outer"
        )
        merged_count = pd.merge(
            global_count_per_hour, suspect_count_per_hour, on="hour_bin", how="outer"
        )

        # If a suspect had no trades in a given hour, suspect_avg_volume or suspect_trade_count might be NaN.
        merged_volume.fillna(0, inplace=True)
        merged_count.fillna(0, inplace=True)

        # 6) Plot: Volume Over Time
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
                name=f"Suspect {selected_suspect} Avg Volume",
            )
        )
        fig_vol.update_layout(
            title=f"Volume Over Time (Hourly Bin) - Trader {selected_suspect}",
            xaxis_title="Time (Hour)",
            yaxis_title="Average Volume",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # 7) Plot: Trade Count Over Time
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
                name=f"Suspect {selected_suspect} Trade Count",
            )
        )
        fig_cnt.update_layout(
            title=f"Trade Count Over Time (Hourly Bin) - Trader {selected_suspect}",
            xaxis_title="Time (Hour)",
            yaxis_title="Number of Trades",
        )
        st.plotly_chart(fig_cnt, use_container_width=True)

        st.markdown(
            f"""
        **Interpretation**: If Trader {selected_suspect} consistently has higher 
        (or spikier) volume or trade frequency than the global average around the same times, 
        that might indicate an anomaly or suspicious behavior.
        """
        )
