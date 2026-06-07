import os

import dlt
import polars as pl

try:
    from .download_data import main as download_all_data
except ImportError:
    from download_data import main as download_all_data


@dlt.source(name="olist_source")
def olist_source(data_dir="data/raw"):
    files = {
        "olist_customers_dataset.csv": "customers",
        "olist_order_items_dataset.csv": "order_items",
        "olist_order_payments_dataset.csv": "order_payments",
        "olist_order_reviews_dataset.csv": "order_reviews",
        "olist_orders_dataset.csv": "orders",
        "olist_products_dataset.csv": "products",
        "olist_sellers_dataset.csv": "sellers",
        "product_category_name_translation.csv": "product_category_name_translation",
    }

    for filename, table_name in files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):

            def make_resource_func(path):
                def get_data():
                    df = pl.read_csv(path)
                    yield df

                return get_data

            yield dlt.resource(
                make_resource_func(file_path),
                name=table_name,
                write_disposition="replace",
            )
        else:
            print(
                f"Warning: file {file_path} not found. Skipping resource '{table_name}'."
            )


def run_pipeline():
    # Download the Olist datasets
    download_all_data()

    # Setup dlt pipeline
    # The database file is located in data/dev.duckdb
    db_path = os.path.abspath("data/dev.duckdb")

    pipeline = dlt.pipeline(
        pipeline_name="olist_pipeline",
        destination="duckdb",
        dataset_name="raw",  # loads data into the 'raw' schema
    )

    print("Running dlt pipeline to load raw Olist data into DuckDB...")
    load_info = pipeline.run(olist_source(), credentials=f"duckdb:///{db_path}")
    print("dlt Load Info:")
    print(load_info)


if __name__ == "__main__":
    run_pipeline()
