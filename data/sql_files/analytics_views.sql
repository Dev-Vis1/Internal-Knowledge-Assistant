-- Northstar Meridian Group analytics views

CREATE VIEW vw_support_case_summary AS
SELECT
    region,
    severity,
    COUNT(*) AS total_cases,
    AVG(first_response_minutes) AS avg_first_response_minutes
FROM support_cases
GROUP BY region, severity;

CREATE VIEW vw_monthly_revenue_health AS
SELECT
    month,
    recurring_revenue_usd,
    churn_revenue_usd,
    expansion_revenue_usd,
    (recurring_revenue_usd - churn_revenue_usd + expansion_revenue_usd) AS net_revenue_delta
FROM finance_monthly_metrics;

CREATE VIEW vw_security_patch_compliance AS
SELECT
    business_unit,
    COUNT(*) AS assets_total,
    SUM(CASE WHEN patch_status = 'Compliant' THEN 1 ELSE 0 END) AS assets_compliant
FROM asset_patch_inventory
GROUP BY business_unit;
