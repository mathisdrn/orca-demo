# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Olist Delivery Delay - Conformalized Quantile Regression (CQR) Training
#
# This script trains a quantile regression model and applies **Conformalized Quantile Regression (CQR)**
# to predict the actual delivery duration (in days) with a 90% prediction interval.
#
# It uses data from `main_marts.fct_orders` and `main_marts.dim_customers`.

# %%
import joblib
import numpy as np
import polars as pl

# Import local analysis configurations and metrics
from config import MODELS_DIR, get_training_data
from metrics import coverage, mae, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Set Polars display configuration
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_float_precision(2)

# %% [markdown]
# ## 1. Load and Clean Data
#
# Load the dataset from the database using our config loader.

# %%
df = get_training_data(use_cache=True)
print(f"Loaded dataset: {df.height} rows, {df.width} columns")

# Extract temporal features from the purchase timestamp
df = df.with_columns(
    purchase_month=pl.col("order_purchase_timestamp").dt.month().cast(pl.Int32),
    purchase_day_of_week=pl.col("order_purchase_timestamp").dt.weekday().cast(pl.Int32),
    purchase_hour=pl.col("order_purchase_timestamp").dt.hour().cast(pl.Int32),
)

df.head()

# %% [markdown]
# ## 2. Feature Selection & Train-Test-Calibration Split
#
# Define target, numeric, and categorical features.

# %%
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

# Extract vectors
X = df.select(features)
y = df.get_column(target)

# Proper train-calibration-test split (60% / 20% / 20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train set size: {X_train.shape[0]} rows")
print(f"Calibration set size: {X_calib.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

# %% [markdown]
# ## 3. Model Definition & Pipeline
#
# We use scikit-learn's `ColumnTransformer` with `OrdinalEncoder` for the categorical variables,
# allowing us to feed native categorical inputs into the `HistGradientBoostingRegressor`.
#
# We set up three separate pipelines:
# - Quantile 0.05 (Lower Bound)
# - Quantile 0.50 (Median Prediction)
# - Quantile 0.95 (Upper Bound)
#
# This produces a 90% confidence target interval ($\alpha = 0.1$).

# %%
alpha = 0.1
q_low = alpha / 2
q_high = 1 - alpha / 2

# Category encoding transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_features,
        )
    ],
    remainder="passthrough",
)


# Base estimator definition
def create_model_pipeline(quantile: float) -> Pipeline:
    regressor = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        categorical_features=[0],  # customer_state is index 0 after transformation
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        early_stopping=True,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


model_low = create_model_pipeline(q_low)
model_median = create_model_pipeline(0.50)
model_high = create_model_pipeline(q_high)

# %% [markdown]
# ## 4. Model Training
#
# Fit the models on the proper training set.

# %%
print("Training model for lower bound (q=0.05)...")
model_low.fit(X_train.to_pandas(), y_train.to_numpy())

print("Training model for median prediction (q=0.50)...")
model_median.fit(X_train.to_pandas(), y_train.to_numpy())

print("Training model for upper bound (q=0.95)...")
model_high.fit(X_train.to_pandas(), y_train.to_numpy())

# %% [markdown]
# ## 5. Conformal Calibration (CQR)
#
# Compute conformity scores on the calibration set to calculate the correction factor $\hat{q}$.

# %%
print("Running conformal calibration...")
cal_low_pred = model_low.predict(X_calib.to_pandas())
cal_high_pred = model_high.predict(X_calib.to_pandas())

# Conformity scores: max(q_low - y, y - q_high)
nonconformity_scores = np.maximum(
    cal_low_pred - y_calib.to_numpy(), y_calib.to_numpy() - cal_high_pred
)

# Compute (1 - alpha) quantile of conformity scores
q_level = np.ceil((len(y_calib) + 1) * (1 - alpha)) / len(y_calib)
qhat = np.quantile(nonconformity_scores, q_level, method="linear")

print(
    f"CQR calibration qhat (alpha={alpha:.1f}): {qhat:.3f} days at quantile level {q_level:.4f}"
)

# %% [markdown]
# ## 6. Evaluate Model on Test Set
#
# Calculate model metrics for both uncalibrated Quantile Regression (QR) and Conformalized Quantile Regression (CQR).

# %%
test_low_pred = model_low.predict(X_test.to_pandas())
test_median_pred = model_median.predict(X_test.to_pandas())
test_high_pred = model_high.predict(X_test.to_pandas())

df_results = pl.DataFrame(
    {
        "True Duration": y_test,
        "Predicted Duration": test_median_pred,
        "Lower Bound": test_low_pred,
        "Upper Bound": test_high_pred,
        "Lower Bound CQR": test_low_pred - qhat,
        "Upper Bound CQR": test_high_pred + qhat,
    }
).with_columns(
    (pl.col("Upper Bound") - pl.col("Lower Bound")).alias("QR Width"),
    (pl.col("Upper Bound CQR") - pl.col("Lower Bound CQR")).alias("CQR Width"),
)

# Compute performance metrics
metrics_comparison = df_results.select(
    # Median performance
    mae("True Duration", "Predicted Duration").alias("MAE Median"),
    r2_score("True Duration", "Predicted Duration").alias("R2 Median (%)"),
    # Coverage
    coverage("Lower Bound", "Upper Bound", "True Duration").alias("QR Coverage"),
    coverage("Lower Bound CQR", "Upper Bound CQR", "True Duration").alias(
        "CQR Coverage"
    ),
    # Widths
    pl.col("QR Width").mean().round(1).alias("Avg QR Width (days)"),
    pl.col("CQR Width").mean().round(1).alias("Avg CQR Width (days)"),
)

print("\nModel Evaluation Summary on Test Set:")
print(metrics_comparison)

# %% [markdown]
# ## 7. Save Models and Conformal Metadata
#
# Save the trained pipelines and the computed calibration margin `qhat` to the models folder.

# %%
print(f"Saving models to {MODELS_DIR}...")
joblib.dump(model_low, MODELS_DIR / "model_low.joblib")
joblib.dump(model_median, MODELS_DIR / "model_median.joblib")
joblib.dump(model_high, MODELS_DIR / "model_high.joblib")

# Save calibration metadata
metadata = {
    "alpha": alpha,
    "qhat": qhat,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "features": features,
    "target": target,
}
joblib.dump(metadata, MODELS_DIR / "metadata.joblib")
print("All artifacts saved successfully!")
