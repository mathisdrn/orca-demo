from dagster import multi_asset, AssetSpec, AssetExecutionContext
from ingestion.olist_dlt_pipeline import run_pipeline


@multi_asset(
    specs=[
        AssetSpec(key=["raw", "customers"], group_name="raw"),
        AssetSpec(key=["raw", "order_items"], group_name="raw"),
        AssetSpec(key=["raw", "order_payments"], group_name="raw"),
        AssetSpec(key=["raw", "order_reviews"], group_name="raw"),
        AssetSpec(key=["raw", "orders"], group_name="raw"),
        AssetSpec(key=["raw", "products"], group_name="raw"),
        AssetSpec(key=["raw", "sellers"], group_name="raw"),
        AssetSpec(key=["raw", "product_category_name_translation"], group_name="raw"),
    ],
    compute_kind="dlt",
)
def olist_raw_data(context: AssetExecutionContext):
    context.log.info("Starting Olist raw data ingestion via dlt...")
    # This runs the dlt pipeline which loads all tables into DuckDB
    run_pipeline()
    context.log.info("Olist raw data ingestion completed.")
