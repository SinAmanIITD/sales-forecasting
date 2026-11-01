-- Rossmann Sales Analysis — SQL layer
-- Run against the `sales` table (cleaned, store-merged train.csv) loaded by
-- src/sql_analysis.py.

-- 1. Weekly sales trend (used to sanity-check seasonality before modeling)
SELECT
    strftime('%Y-%W', Date) AS year_week,
    SUM(Sales)              AS weekly_sales,
    SUM(Customers)          AS weekly_customers,
    ROUND(SUM(Sales) * 1.0 / NULLIF(SUM(Customers), 0), 2) AS sales_per_customer
FROM sales
GROUP BY year_week
ORDER BY year_week;

-- 2. Promo impact on average sales, by store type
SELECT
    StoreType,
    Promo,
    COUNT(*)            AS n_records,
    ROUND(AVG(Sales), 2) AS avg_sales
FROM sales
GROUP BY StoreType, Promo
ORDER BY StoreType, Promo;

-- 3. Holiday impact on sales
SELECT
    StateHoliday,
    SchoolHoliday,
    COUNT(*)            AS n_records,
    ROUND(AVG(Sales), 2) AS avg_sales
FROM sales
GROUP BY StateHoliday, SchoolHoliday
ORDER BY avg_sales DESC;

-- 4. Day-of-week seasonality
SELECT
    DayOfWeek,
    ROUND(AVG(Sales), 2)      AS avg_sales,
    ROUND(AVG(Customers), 2)  AS avg_customers
FROM sales
GROUP BY DayOfWeek
ORDER BY DayOfWeek;

-- 5. Top 20 stores by average daily sales (feeds the "Store forecast" page's
--    default store list)
SELECT
    Store,
    StoreType,
    Assortment,
    COUNT(*)              AS n_days,
    ROUND(AVG(Sales), 2)   AS avg_sales,
    ROUND(SUM(Sales), 2)   AS total_sales
FROM sales
GROUP BY Store
ORDER BY avg_sales DESC
LIMIT 20;

-- 6. Competition distance bucket vs. average sales — tests whether nearby
--    competition measurably depresses sales (informs the
--    CompetitionDistance feature's inclusion in the model)
SELECT
    CASE
        WHEN CompetitionDistance < 500   THEN '0-500m'
        WHEN CompetitionDistance < 2000  THEN '500m-2km'
        WHEN CompetitionDistance < 10000 THEN '2km-10km'
        ELSE '10km+'
    END AS competition_bucket,
    COUNT(DISTINCT Store)  AS n_stores,
    ROUND(AVG(Sales), 2)    AS avg_sales
FROM sales
GROUP BY competition_bucket
ORDER BY MIN(CompetitionDistance);

-- 7. Monthly seasonality (feeds the "Distribution" tab's seasonality chart)
SELECT
    CAST(strftime('%m', Date) AS INTEGER) AS month,
    ROUND(AVG(Sales), 2) AS avg_sales
FROM sales
GROUP BY month
ORDER BY month;
