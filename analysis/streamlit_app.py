import altair as alt
import joblib
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# Import configuration
from config import MODELS_DIR, get_training_data
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Olist Delivery Performance & Conformal Prediction",
    page_icon="🚚",
    layout="wide",
)

# App Title
st.title("🚚 Olist Delivery Performance & Conformal Prediction")
st.markdown("""
This dashboard analyzes the delivery performance of the Brazilian E-Commerce (Olist) dataset 
and provides certified **conformal prediction intervals** for order delivery durations.
""")

# Model paths and availability check
model_low_path = MODELS_DIR / "model_low.joblib"
model_median_path = MODELS_DIR / "model_median.joblib"
model_high_path = MODELS_DIR / "model_high.joblib"
metadata_path = MODELS_DIR / "metadata.joblib"

models_available = (
    model_low_path.exists()
    and model_median_path.exists()
    and model_high_path.exists()
    and metadata_path.exists()
)


# Load data helper
@st.cache_data
def load_data():
    df = get_training_data(use_cache=True)
    # Add temporal columns
    df = df.with_columns(
        purchase_year=pl.col("order_purchase_timestamp").dt.year().cast(pl.Int32),
        purchase_month=pl.col("order_purchase_timestamp").dt.month().cast(pl.Int32),
        purchase_month_str=pl.col("order_purchase_timestamp").dt.strftime("%Y-%m"),
        is_late=pl.col("actual_delivery_duration_days")
        > pl.col("estimated_delivery_duration_days"),
    )
    return df


@st.cache_data
def get_test_predictions():
    if not models_available:
        return None

    # Load raw data and extract same features as model_training
    df = get_training_data(use_cache=True)
    df = df.with_columns(
        purchase_month=pl.col("order_purchase_timestamp").dt.month().cast(pl.Int32),
        purchase_day_of_week=pl.col("order_purchase_timestamp")
        .dt.weekday()
        .cast(pl.Int32),
        purchase_hour=pl.col("order_purchase_timestamp").dt.hour().cast(pl.Int32),
    )

    target = "actual_delivery_duration_days"
    categorical_features = ["customer_state"]
    numerical_features = [
        "estimated_delivery_duration_days",
        "total_items_count",
        "total_price",
        "total_freight_value",
        "total_order_value",
        "total_payment_value",
        "max_payment_installments",
        "avg_review_score",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour",
    ]
    features = categorical_features + numerical_features
    X = df.select(features)
    y = df.get_column(target)

    # Split
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Predict
    model_low = joblib.load(model_low_path)
    model_median = joblib.load(model_median_path)
    model_high = joblib.load(model_high_path)
    metadata = joblib.load(metadata_path)
    qhat = metadata["qhat"]

    pred_low = model_low.predict(X_test.to_pandas())
    pred_med = model_median.predict(X_test.to_pandas())
    pred_high = model_high.predict(X_test.to_pandas())

    cal_low_pred = model_low.predict(X_calib.to_pandas())
    cal_high_pred = model_high.predict(X_calib.to_pandas())
    conformity_scores = np.maximum(
        cal_low_pred - y_calib.to_numpy(), y_calib.to_numpy() - cal_high_pred
    )

    return {
        "y_test": y_test.to_numpy(),
        "pred_low": pred_low,
        "pred_med": pred_med,
        "pred_high": pred_high,
        "qhat": qhat,
        "conformity_scores": conformity_scores,
        "estimated_duration": X_test.get_column(
            "estimated_delivery_duration_days"
        ).to_numpy(),
    }


try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data from database: {e}")
    data_loaded = False


# Percentage change helper
def percentage_change(current, previous):
    if previous == 0 or previous is None:
        return 0.0
    return (current - previous) / previous


# ----------------------------------------------------
# Main Content
# ----------------------------------------------------
if data_loaded:
    # --- Metrics Section (Year on Year) ---
    st.header("Overview - Month on Month (Recent)")

    # Calculate MoM metrics using latest month and previous month in dataset
    metrics_df = (
        df.group_by("purchase_month_str")
        .agg(
            order_count=pl.len(),
            avg_duration=pl.col("actual_delivery_duration_days").mean(),
            late_rate=pl.col("is_late").mean() * 100,
        )
        .sort("purchase_month_str", descending=True)
    )

    if metrics_df.height >= 2:
        current_month = metrics_df.row(0, named=True)
        previous_month = metrics_df.row(1, named=True)

        col1, col2, col3 = st.columns(3)

        # Order Count Metric
        col1.metric(
            label=f"Orders in {current_month['purchase_month_str']}",
            value=f"{current_month['order_count']:,} orders",
            delta=f"{percentage_change(current_month['order_count'], previous_month['order_count']):.1%}",
        )

        # Average Delivery Time Metric
        col2.metric(
            label="Avg Delivery Time",
            value=f"{current_month['avg_duration']:.1f} days",
            delta=f"{current_month['avg_duration'] - previous_month['avg_duration']:.2f} days",
            delta_color="inverse",
        )

        # Late Delivery Rate Metric
        col3.metric(
            label="Late Delivery Rate",
            value=f"{current_month['late_rate']:.1f}%",
            delta=f"{current_month['late_rate'] - previous_month['late_rate']:.2f}%",
            delta_color="inverse",
        )
    else:
        st.info("Insufficient historical month-on-month data to compute comparisons.")

    # --- Plotting Section ---
    st.header("Insights & Analytics")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Delivery Time Analysis",
            "📈 Trends Over Time",
            "😊 Customer Satisfaction",
            "🌍 Geographic Logistics",
            "📊 Model Calibration & Diagnostics",
        ]
    )

    with tab1:
        col_left, col_right = st.columns(2)

        with col_left:
            # Distribution of delivery times
            duration_dist = (
                df.filter(pl.col("actual_delivery_duration_days") <= 45)
                .group_by("actual_delivery_duration_days")
                .agg(order_count=pl.len())
                .sort("actual_delivery_duration_days")
            )

            chart_dist = (
                alt.Chart(duration_dist)
                .mark_bar(color="#3182bd")
                .encode(
                    x=alt.X(
                        "actual_delivery_duration_days:Q",
                        title="Delivery Duration (Days)",
                    ),
                    y=alt.Y("order_count:Q", title="Number of Orders"),
                    tooltip=[
                        alt.Tooltip("actual_delivery_duration_days:Q", title="Days"),
                        alt.Tooltip("order_count:Q", title="Orders"),
                    ],
                )
                .properties(
                    title="Distribution of Delivery Durations (up to 45 days)",
                    height=350,
                )
            )
            st.altair_chart(chart_dist, width="stretch")

        with col_right:
            # Avg delivery time by customer state
            state_delivery = (
                df.group_by("customer_state")
                .agg(
                    avg_duration=pl.col("actual_delivery_duration_days").mean(),
                    order_count=pl.len(),
                )
                .filter(pl.col("order_count") > 100)  # Filter rare states
                .sort("avg_duration", descending=True)
            )

            chart_state = (
                alt.Chart(state_delivery)
                .mark_bar(color="#e6550d")
                .encode(
                    x=alt.X("avg_duration:Q", title="Average Delivery Time (Days)"),
                    y=alt.Y("customer_state:N", sort="-x", title="Customer State"),
                    tooltip=[
                        alt.Tooltip("customer_state:N", title="State"),
                        alt.Tooltip("avg_duration:Q", title="Avg Days", format=".1f"),
                        alt.Tooltip("order_count:Q", title="Total Orders"),
                    ],
                )
                .properties(title="Average Delivery Duration by State", height=350)
            )
            st.altair_chart(chart_state, width="stretch")

    with tab2:
        # Trends over time (group by month string)
        trends = (
            df.group_by("purchase_month_str")
            .agg(
                order_count=pl.len(),
                avg_actual=pl.col("actual_delivery_duration_days").mean(),
                avg_estimated=pl.col("estimated_delivery_duration_days").mean(),
            )
            .sort("purchase_month_str")
        )

        # Select every second month for axis labels to avoid overlap
        axis_values = trends.get_column("purchase_month_str").to_list()[::2]

        chart_orders_trend = (
            alt.Chart(trends)
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=alt.X(
                    "purchase_month_str:O",
                    title="Purchase Month",
                    axis=alt.Axis(values=axis_values),
                ),
                y=alt.Y("order_count:Q", title="Order Count"),
                tooltip=["purchase_month_str", "order_count"],
            )
            .properties(title="Monthly Order Volume Trend", height=350)
        )
        st.altair_chart(chart_orders_trend, width="stretch")

    with tab3:
        st.subheader("😊 Delivery Delay vs. Customer Satisfaction")
        st.markdown("""
        How does delivery delay affect customer satisfaction? Below, we analyze the average review score 
        based on how many days early or late the delivery was relative to the estimate.
        """)

        # Calculate delivery delay
        sat_df = df.with_columns(
            delay=(
                pl.col("actual_delivery_duration_days")
                - pl.col("estimated_delivery_duration_days")
            ).cast(pl.Int32)
        )

        # Classify delays into categories
        sat_cat_df = sat_df.with_columns(
            delay_cat=pl.when(pl.col("delay") < 0)
            .then(pl.lit("1. Early / On Time"))
            .when(pl.col("delay") == 0)
            .then(pl.lit("2. Exact Day"))
            .when(pl.col("delay") <= 3)
            .then(pl.lit("3. 1-3 Days Late"))
            .when(pl.col("delay") <= 7)
            .then(pl.lit("4. 4-7 Days Late"))
            .otherwise(pl.lit("5. >7 Days Late"))
        )

        # Aggregate review scores
        delay_cat_summary = (
            sat_cat_df.group_by("delay_cat")
            .agg(avg_score=pl.col("avg_review_score").mean(), order_count=pl.len())
            .sort("delay_cat")
        )

        # Create columns for details
        col_sat_l, col_sat_r = st.columns(2)

        with col_sat_l:
            chart_sat_cat = (
                alt.Chart(delay_cat_summary)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "delay_cat:N",
                        title="Delivery Status",
                        axis=alt.Axis(labelAngle=-15),
                    ),
                    y=alt.Y(
                        "avg_score:Q",
                        title="Average Review Score (1-5)",
                        scale=alt.Scale(domain=[1, 5]),
                    ),
                    color=alt.Color(
                        "delay_cat:N",
                        scale=alt.Scale(
                            domain=[
                                "1. Early / On Time",
                                "2. Exact Day",
                                "3. 1-3 Days Late",
                                "4. 4-7 Days Late",
                                "5. >7 Days Late",
                            ],
                            range=[
                                "#2ca02c",
                                "#bcbd22",
                                "#ff7f0e",
                                "#d62728",
                                "#8c564b",
                            ],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("delay_cat:N", title="Category"),
                        alt.Tooltip(
                            "avg_score:Q", title="Avg Review Score", format=".2f"
                        ),
                        alt.Tooltip("order_count:Q", title="Orders", format=","),
                    ],
                )
                .properties(
                    title="Average Customer Review Score by Delivery Status", height=350
                )
            )
            st.altair_chart(chart_sat_cat, width="stretch")

        with col_sat_r:
            # Line/Area chart of satisfaction vs precise delay days (-10 to +20 days)
            delay_trend = (
                sat_df.filter((pl.col("delay") >= -10) & (pl.col("delay") <= 20))
                .group_by("delay")
                .agg(avg_score=pl.col("avg_review_score").mean(), order_count=pl.len())
                .sort("delay")
            )

            chart_sat_trend = (
                alt.Chart(delay_trend)
                .mark_line(point=True, color="#1f77b4")
                .encode(
                    x=alt.X(
                        "delay:Q",
                        title="Delivery Delay (Days: Negative = Early, Positive = Late)",
                    ),
                    y=alt.Y(
                        "avg_score:Q",
                        title="Average Review Score",
                        scale=alt.Scale(domain=[1, 5]),
                    ),
                    tooltip=[
                        alt.Tooltip("delay:Q", title="Delay Days"),
                        alt.Tooltip(
                            "avg_score:Q", title="Avg Review Score", format=".2f"
                        ),
                        alt.Tooltip("order_count:Q", title="Total Orders", format=","),
                    ],
                )
                .properties(
                    title="Satisfaction Trend by Exact Delay (Days)", height=350
                )
            )
            st.altair_chart(chart_sat_trend, width="stretch")

        st.info("""
        💡 **Key Takeaway**: There is a stark correlation between delivery delay and customer satisfaction. 
        Orders delivered early or on time maintain a high average review score (~4.3 / 5.0). 
        However, even minor delays (1-3 days late) drop satisfaction to ~3.3, and severe delays (>7 days late) 
        lead to a bottom-tier rating of ~1.7. This highlights the business value of accurate delivery estimations 
        and reliable logistics.
        """)

    with tab4:
        st.subheader("🌍 Geographic Logistics & Cost Correlation")
        st.markdown("""
        How do shipping fees and delivery times vary by customer state? This interactive bubble chart 
        correlates freight prices, delivery durations, and late rates across Brazil.
        """)

        # Aggregate by state
        geo_summary = (
            df.group_by("customer_state")
            .agg(
                avg_freight=pl.col("total_freight_value").mean(),
                avg_duration=pl.col("actual_delivery_duration_days").mean(),
                late_rate=(pl.col("is_late").cast(pl.Float64).mean() * 100),
                order_count=pl.len(),
            )
            .filter(pl.col("order_count") > 50)  # Filter states with negligible volume
            .sort("avg_duration")
        )

        # Bubble chart
        chart_geo_bubble = (
            alt.Chart(geo_summary)
            .mark_circle()
            .encode(
                x=alt.X(
                    "avg_duration:Q",
                    title="Average Delivery Duration (Days)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "avg_freight:Q",
                    title="Average Freight Value (R$)",
                    scale=alt.Scale(zero=False),
                ),
                size=alt.Size(
                    "order_count:Q",
                    title="Order Volume",
                    scale=alt.Scale(range=[100, 1000]),
                ),
                color=alt.Color(
                    "late_rate:Q", title="Late Rate (%)", scale=alt.Scale(scheme="reds")
                ),
                tooltip=[
                    alt.Tooltip("customer_state:N", title="State"),
                    alt.Tooltip(
                        "avg_duration:Q", title="Avg Duration (Days)", format=".1f"
                    ),
                    alt.Tooltip(
                        "avg_freight:Q", title="Avg Freight Cost (R$)", format=".2f"
                    ),
                    alt.Tooltip(
                        "late_rate:Q", title="Late Delivery Rate", format=".1f"
                    ),
                    alt.Tooltip("order_count:Q", title="Total Orders", format=","),
                ],
            )
            .properties(
                title="State-wise Shipping Fees vs. Delivery Durations", height=450
            )
            .interactive()
        )

        st.altair_chart(chart_geo_bubble, width="stretch")

        st.info("""
        💡 **How to read this chart**:
        - **Position**: Further right means slower delivery; further up means more expensive freight.
        - **Size**: Bigger bubbles represent states with higher order volume (e.g., SP, RJ, MG).
        - **Color Intensity**: Darker red circles indicate higher late delivery rates.
        - **Insight**: States in the North/Northeast (like PB, AL, PA, MA) suffer a double penalty of high shipping costs 
          (R$ 40+) and slow transit times (20-25 days), whereas the South/Southeast regions (like SP, PR, MG) enjoy cheap 
          (R$ 15-20) and fast (5-11 days) deliveries.
        """)

    with tab5:
        st.subheader("📊 Conformal Prediction Calibration & Diagnostics")

        if not models_available:
            st.warning(
                "⚠️ Machine learning models are not trained yet. Cannot display diagnostics."
            )
        else:
            with st.spinner("Evaluating models on test set..."):
                test_data = get_test_predictions()

            if test_data is None:
                st.error("Error evaluating test predictions.")
            else:
                y_t = test_data["y_test"]
                pred_med = test_data["pred_med"]
                pred_low = test_data["pred_low"]
                pred_high = test_data["pred_high"]
                qhat = test_data["qhat"]
                conf_scores = test_data["conformity_scores"]
                est_durations = test_data["estimated_duration"]

                # Conformal boundaries
                cqr_low = np.maximum(0, pred_low - qhat)
                cqr_high = pred_high + qhat

                # Calculations
                mae_val = np.mean(np.abs(y_t - pred_med))
                r2_val = 1 - (
                    np.sum((y_t - pred_med) ** 2) / np.sum((y_t - np.mean(y_t)) ** 2)
                )
                coverage_val = np.mean((y_t >= cqr_low) & (y_t <= cqr_high)) * 100
                avg_width_val = np.mean(cqr_high - cqr_low)

                # Metrics layout
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Test MAE", f"{mae_val:.2f} days")
                col_m2.metric("Test R²", f"{r2_val:.1%}")
                col_m3.metric("Empirical Coverage", f"{coverage_val:.1f}%")
                col_m4.metric("Avg Interval Width", f"{avg_width_val:.1f} days")

                # Plot 1: Conformity Score Distribution
                scores_df = pl.DataFrame({"conformity_score": conf_scores})
                p99 = np.percentile(conf_scores, 99)
                scores_filtered = scores_df.filter(pl.col("conformity_score") <= p99)

                chart_conf_dist = (
                    alt.Chart(scores_filtered)
                    .mark_bar(opacity=0.6, color="#4c78a8")
                    .encode(
                        x=alt.X(
                            "conformity_score:Q",
                            bin=alt.Bin(maxbins=40),
                            title="Conformity Score (Days)",
                        ),
                        y=alt.Y("count()", title="Count of Calibration Samples"),
                    )
                )

                qhat_df = pl.DataFrame({"qhat": [qhat]})
                rule = (
                    alt.Chart(qhat_df)
                    .mark_rule(color="#e15759", strokeWidth=2, strokeDash=[5, 5])
                    .encode(x="qhat:Q")
                )

                text = (
                    alt.Chart(qhat_df)
                    .mark_text(
                        align="left", dx=5, dy=-150, color="#e15759", fontWeight="bold"
                    )
                    .encode(
                        x="qhat:Q", text=alt.value(f"q-hat threshold = {qhat:.2f} days")
                    )
                )

                chart_calibration = (chart_conf_dist + rule + text).properties(
                    title="Conformity Scores Distribution (Calibration Set)", height=350
                )

                # Plot 2: Adaptive Width vs. Difficulty
                width_df = pl.DataFrame(
                    {
                        "estimated_duration": est_durations,
                        "interval_width": cqr_high - cqr_low,
                    }
                )

                width_summary = (
                    width_df.group_by("estimated_duration")
                    .agg(avg_width=pl.col("interval_width").mean(), count=pl.len())
                    .filter(pl.col("count") > 10)
                    .sort("estimated_duration")
                )

                chart_adaptive_width = (
                    alt.Chart(width_summary)
                    .mark_line(point=True, color="#f28e2b")
                    .encode(
                        x=alt.X(
                            "estimated_duration:Q",
                            title="Estimated Delivery Duration (Days)",
                        ),
                        y=alt.Y(
                            "avg_width:Q",
                            title="Average Prediction Interval Width (Days)",
                        ),
                        tooltip=[
                            alt.Tooltip("estimated_duration:Q", title="Estimated Days"),
                            alt.Tooltip(
                                "avg_width:Q", title="Avg Interval Width", format=".1f"
                            ),
                            alt.Tooltip("count:Q", title="Sample Count"),
                        ],
                    )
                    .properties(
                        title="CQR Interval Width vs. Estimated Duration", height=350
                    )
                )

                col_diag_l, col_diag_r = st.columns(2)
                with col_diag_l:
                    st.altair_chart(chart_calibration, width="stretch")
                with col_diag_r:
                    st.altair_chart(chart_adaptive_width, width="stretch")

                st.info(r"""
                💡 **Understanding Conformalized Quantile Regression (CQR)**:
                - **Why coverage is exactly 90%**: Standard machine learning models often over- or under-predict uncertainty. Conformal prediction solves this by looking at how wrong the model was on a hold-out **calibration set**. The safety margin $\hat{q}$ is chosen such that exactly $1 - \alpha = 90\%$ of the calibration errors fall below it.
                - **Why intervals are adaptive**: Traditional conformal prediction adds a constant band (e.g. $\pm 5$ days) to every prediction. CQR instead conforms the *quantiles* of a quantile regression model. This means that for orders with long estimated durations (which are inherently harder to predict), the interval naturally expands, while for short local shipments, the interval shrinks. This is demonstrated by the rising line on the right!
                """)

    # --- Conformal Prediction Section ---
    st.write("---")
    st.header("🔮 Interactive Conformal Delivery Predictor")
    st.markdown("""
    Predict the exact delivery duration of an order. Conformal prediction provides a **90% coverage guarantee**, 
    meaning the true delivery duration will fall inside the interval exactly 90% of the time on average.
    """)

    # Load conformal model and metadata
    model_low_path = MODELS_DIR / "model_low.joblib"
    model_median_path = MODELS_DIR / "model_median.joblib"
    model_high_path = MODELS_DIR / "model_high.joblib"
    metadata_path = MODELS_DIR / "metadata.joblib"

    if not (
        model_low_path.exists()
        and model_median_path.exists()
        and model_high_path.exists()
        and metadata_path.exists()
    ):
        st.warning(
            "⚠️ Machine learning models are not trained yet. Please trigger model training in the Dagster UI or run `python analysis/model_training.py`."
        )
    else:
        # Load pipelines and calibration parameters
        model_low = joblib.load(model_low_path)
        model_median = joblib.load(model_median_path)
        model_high = joblib.load(model_high_path)
        metadata = joblib.load(metadata_path)
        qhat = metadata["qhat"]

        # User input form
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            customer_state = st.selectbox(
                "Customer State (destination)",
                options=sorted(df.get_column("customer_state").unique().to_list()),
                index=sorted(df.get_column("customer_state").unique().to_list()).index(
                    "SP"
                )
                if "SP" in df.get_column("customer_state").unique()
                else 0,
            )
            estimated_duration = st.slider(
                "Estimated Delivery Duration (Days)",
                min_value=1,
                max_value=60,
                value=15,
            )

        with col_in2:
            items_count = st.number_input(
                "Total Items in Order", min_value=1, max_value=20, value=1
            )
            total_price = st.number_input(
                "Total Order Value (R$)", min_value=5.0, max_value=5000.0, value=120.0
            )
            total_freight = st.number_input(
                "Freight Cost (R$)", min_value=0.0, max_value=500.0, value=18.5
            )

        with col_in3:
            purchase_month = st.slider(
                "Purchase Month", min_value=1, max_value=12, value=6
            )
            # Map week day
            weekdays = {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
            }
            day_name = st.selectbox(
                "Purchase Day of Week", options=list(weekdays.values()), index=0
            )
            purchase_day = [k for k, v in weekdays.items() if v == day_name][0]
            purchase_hour = st.slider(
                "Purchase Hour", min_value=0, max_value=23, value=14
            )

        # Define input dict before the button to make it accessible to simulator
        input_dict = {
            "customer_state": [customer_state],
            "estimated_delivery_duration_days": [estimated_duration],
            "total_items_count": [items_count],
            "total_price": [total_price],
            "total_freight_value": [total_freight],
            "total_order_value": [total_price + total_freight],
            "total_payment_value": [total_price + total_freight],
            "max_payment_installments": [1],
            "avg_review_score": [5.0],
            "purchase_month": [purchase_month],
            "purchase_day_of_week": [purchase_day],
            "purchase_hour": [purchase_hour],
        }

        if st.button("🔮 Predict Delivery Duration", type="primary"):
            input_df = pd.DataFrame(input_dict)

            # Predict bounds
            pred_low = model_low.predict(input_df)[0]
            pred_median = model_median.predict(input_df)[0]
            pred_high = model_high.predict(input_df)[0]

            # Conformal intervals
            cqr_low = max(0.0, pred_low - qhat)
            cqr_high = pred_high + qhat

            st.success("### Prediction Results")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="Estimated Median Delivery Duration",
                    value=f"{pred_median:.1f} days",
                )

            with res_col2:
                st.metric(
                    label="90% Conformal Prediction Interval",
                    value=f"{cqr_low:.1f} to {cqr_high:.1f} days",
                )

            st.info(rf"""
            **What does this mean?**
            - The model predicts that this package will most likely arrive in **{pred_median:.1f} days**.
            - Conformal theory guarantees that there is a **90% probability** that the package will arrive in **{cqr_low:.1f} to {cqr_high:.1f} days**.
            - The conformal margin ($\hat{{q}}$) applied is **{qhat:.3f} days**.
            """)

        # --- What-if Sensitivity Simulator ---
        st.write("---")
        with st.expander("📈 What-If Sensitivity Simulator (Live)", expanded=True):
            st.markdown("""
            See how changing inputs affects the predicted delivery duration and the 90% confidence interval.
            Choose a feature to vary while holding all other inputs constant:
            """)

            sim_feature = st.selectbox(
                "Select Feature to Vary",
                options=[
                    "Estimated Delivery Duration (Days)",
                    "Total Order Value (R$)",
                    "Total Items in Order",
                ],
                key="sim_feature_select",
            )

            # Build simulation dataframe
            if sim_feature == "Estimated Delivery Duration (Days)":
                sim_range = np.arange(5, 45, 1)
                sim_inputs = {k: [v[0]] * len(sim_range) for k, v in input_dict.items()}
                sim_inputs["estimated_delivery_duration_days"] = sim_range.tolist()
                x_col = "estimated_delivery_duration_days"
                x_title = "Estimated Delivery Duration (Days)"
            elif sim_feature == "Total Order Value (R$)":
                sim_range = np.linspace(10, 1000, 50)
                sim_inputs = {k: [v[0]] * len(sim_range) for k, v in input_dict.items()}
                sim_inputs["total_price"] = sim_range.tolist()
                sim_inputs["total_order_value"] = (sim_range + total_freight).tolist()
                sim_inputs["total_payment_value"] = (sim_range + total_freight).tolist()
                x_col = "total_price"
                x_title = "Total Price (R$)"
            else:  # Total Items in Order
                sim_range = np.arange(1, 11, 1)
                sim_inputs = {k: [v[0]] * len(sim_range) for k, v in input_dict.items()}
                sim_inputs["total_items_count"] = sim_range.tolist()
                x_col = "total_items_count"
                x_title = "Total Items in Order"

            sim_df = pd.DataFrame(sim_inputs)

            # Predict across the range
            sim_pred_low = model_low.predict(sim_df)
            sim_pred_med = model_median.predict(sim_df)
            sim_pred_high = model_high.predict(sim_df)

            sim_cqr_low = np.maximum(0.0, sim_pred_low - qhat)
            sim_cqr_high = sim_pred_high + qhat

            sim_results = pd.DataFrame(
                {
                    x_col: sim_range,
                    "Median Prediction": sim_pred_med,
                    "Lower Bound (90%)": sim_cqr_low,
                    "Upper Bound (90%)": sim_cqr_high,
                }
            )

            # Create Altair plot with shaded confidence interval
            # Line chart for median
            line_med = (
                alt.Chart(sim_results)
                .mark_line(color="#1f77b4", strokeWidth=2)
                .encode(
                    x=alt.X(f"{x_col}:Q", title=x_title),
                    y=alt.Y("Median Prediction:Q", title="Delivery Duration (Days)"),
                    tooltip=[
                        alt.Tooltip(f"{x_col}:Q", title=x_title, format=".1f"),
                        alt.Tooltip(
                            "Median Prediction:Q", title="Predicted Days", format=".1f"
                        ),
                        alt.Tooltip(
                            "Lower Bound (90%):Q",
                            title="Lower Bound (90%)",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "Upper Bound (90%):Q",
                            title="Upper Bound (90%)",
                            format=".1f",
                        ),
                    ],
                )
            )

            # Shaded area for conformal interval
            area_conf = (
                alt.Chart(sim_results)
                .mark_area(opacity=0.2, color="#1f77b4")
                .encode(
                    x=alt.X(f"{x_col}:Q"),
                    y=alt.Y("Lower Bound (90%):Q"),
                    y2=alt.Y2("Upper Bound (90%):Q"),
                )
            )

            chart_sim = (area_conf + line_med).properties(
                title=f"Sensitivity Analysis: Delivery Prediction vs. {x_title}",
                height=350,
            )

            st.altair_chart(chart_sim, width="stretch")
else:
    st.error(
        "No data found. Ensure dev.duckdb is loaded and has records inside the schemas."
    )
