import os
from pathlib import Path

import duckdb
import polars as pl

# Define paths
ANALYSIS_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = ANALYSIS_DIR.parent
DB_PATH = PROJECT_DIR / "data" / "dev.duckdb"
CACHE_DIR = ANALYSIS_DIR / "cache"
MODELS_DIR = ANALYSIS_DIR / "models"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create and return a connection to the local DuckDB database."""
    # Resolve database path relative to environment if set
    db_path = os.getenv("DBT_DATABASE_PATH", str(DB_PATH))
    return duckdb.connect(db_path, read_only=True)


def get_training_data(*, use_cache: bool = True) -> pl.DataFrame:
    """Retrieve and cache order delivery training dataset from DuckDB.

    Only uses marts schema tables to align with architecture preferences.
    """
    filepath = CACHE_DIR / "training_data.parquet"

    if use_cache and filepath.exists():
        df = pl.read_parquet(filepath)
    else:
        query = """
        SELECT 
            o.order_id,
            o.actual_delivery_duration_days,
            o.estimated_delivery_duration_days,
            o.total_items_count,
            o.total_price,
            o.total_freight_value,
            o.total_order_value,
            o.total_payment_value,
            o.max_payment_installments,
            o.avg_review_score,
            c.customer_state,
            c.customer_city,
            c.customer_zip_code_prefix,
            o.order_purchase_timestamp
        FROM main_marts.fct_orders o
        LEFT JOIN main_marts.dim_customers c 
            ON o.customer_unique_id = c.customer_unique_id
        WHERE o.order_status = 'delivered'
          AND o.actual_delivery_duration_days IS NOT NULL
          AND o.actual_delivery_duration_days >= 0
        """
        conn = get_duckdb_connection()
        try:
            df = conn.execute(query).pl()
        finally:
            conn.close()

        # Cache the dataset as parquet
        df.write_parquet(filepath)

    return df
