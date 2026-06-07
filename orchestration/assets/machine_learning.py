import subprocess
import os
from dagster import asset, AssetKey, AssetExecutionContext


@asset(
    deps=[AssetKey(["marts", "fct_orders"]), AssetKey(["marts", "dim_customers"])],
    group_name="machine_learning",
    compute_kind="python",
    description="Trains the Conformalized Quantile Regression models for delivery delay prediction.",
)
def trained_conformal_model(context: AssetExecutionContext):
    """Retrains the conformal prediction models using the updated marts data."""
    context.log.info("Starting conformal prediction model training...")

    # Resolve the absolute path to the training script
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    script_path = os.path.join(project_dir, "analysis", "model_training.py")

    context.log.info(f"Executing training script: {script_path}")

    # Run the Jupytext python script in the environment
    result = subprocess.run(
        ["uv", "run", "python", script_path],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    # Log stdout and stderr
    context.log.info(result.stdout)
    if result.stderr:
        context.log.warning(result.stderr)

    context.log.info("Model training completed and model joblib files saved.")
