select
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state
from {{ ref('stg_customers') }}
group by 1, 2, 3, 4
