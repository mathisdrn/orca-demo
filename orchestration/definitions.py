from dagster import Definitions
from dagster_dbt import DbtCliResource
from .assets.ingestion import olist_raw_data
from .assets.transformation import olist_dbt_assets, dbt_project
from .assets.machine_learning import trained_conformal_model

defs = Definitions(
    assets=[olist_raw_data, olist_dbt_assets, trained_conformal_model],
    resources={
        "dbt": DbtCliResource(project_dir=dbt_project),
    },
)
