import os
from pathlib import Path
from typing import Any, Mapping, Optional
from dagster import AssetExecutionContext
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, dbt_assets, DbtProject

# Path to the dbt project folder
dbt_project_dir = (
    Path(__file__).parent.parent.parent.joinpath("transformation").resolve()
)

# Ensure the data directory exists so DuckDB path validation passes during parse/compile
project_root = Path(__file__).parent.parent.parent.resolve()
data_dir = project_root.joinpath("data")
data_dir.mkdir(exist_ok=True)

dbt_project = DbtProject(
    project_dir=os.fspath(dbt_project_dir), profiles_dir=os.fspath(dbt_project_dir)
)

# Compiles the dbt project manifest if it is running in development mode
# or if the manifest does not exist yet (e.g. in a newly cloned project)
if not dbt_project.manifest_path.exists():
    dbt_project.preparer.prepare(dbt_project)
else:
    dbt_project.prepare_if_dev()


class CustomDagsterDbtTranslator(DagsterDbtTranslator):
    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> Optional[str]:
        # Extract the subdirectory name from the FQN (e.g. ['olist_analytics', 'staging', 'stg_customers'])
        fqn = dbt_resource_props.get("fqn", [])
        if len(fqn) > 2:
            return fqn[1]  # "staging" or "marts"
        return super().get_group_name(dbt_resource_props)


@dbt_assets(
    manifest=dbt_project.manifest_path,
    dagster_dbt_translator=CustomDagsterDbtTranslator(),
)
def olist_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    # This runs dbt build which runs both models and tests
    yield from dbt.cli(["build"], context=context).stream()
