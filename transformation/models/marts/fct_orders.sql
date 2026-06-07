with orders as (
    select * from {{ ref('stg_orders') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

items as (
    select
        order_id,
        count(product_id) as total_items_count,
        sum(price) as total_price,
        sum(freight_value) as total_freight_value,
        sum(price + freight_value) as total_order_value
    from {{ ref('stg_order_items') }}
    group by 1
),

payments as (
    select
        order_id,
        sum(payment_value) as total_payment_value,
        max(payment_installments) as max_payment_installments
    from {{ ref('stg_order_payments') }}
    group by 1
),

reviews as (
    select
        order_id,
        avg(review_score) as avg_review_score,
        count(review_id) as review_count
    from {{ ref('stg_order_reviews') }}
    group by 1
)

select
    o.order_id,
    c.customer_unique_id,
    o.order_status,
    
    -- Timestamps
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Calculated delivery metrics (in days)
    date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) as actual_delivery_duration_days,
    date_diff('day', o.order_purchase_timestamp, o.order_estimated_delivery_date) as estimated_delivery_duration_days,
    date_diff('day', o.order_delivered_customer_date, o.order_estimated_delivery_date) as delivery_delta_days, -- positive means early, negative means late
    case
        when o.order_delivered_customer_date > o.order_estimated_delivery_date then true
        else false
    end as is_delivery_delayed,

    -- Financial metrics
    coalesce(i.total_items_count, 0) as total_items_count,
    coalesce(i.total_price, 0.0) as total_price,
    coalesce(i.total_freight_value, 0.0) as total_freight_value,
    coalesce(i.total_order_value, 0.0) as total_order_value,
    coalesce(p.total_payment_value, 0.0) as total_payment_value,
    coalesce(p.max_payment_installments, 0) as max_payment_installments,
    
    -- Review metrics
    coalesce(r.avg_review_score, 0.0) as avg_review_score,
    coalesce(r.review_count, 0) as review_count

from orders o
left join customers c
    on o.customer_id = c.customer_id
left join items i
    on o.order_id = i.order_id
left join payments p
    on o.order_id = p.order_id
left join reviews r
    on o.order_id = r.order_id
