import duckdb
import pandas as pd
import numpy as np

DB_PATH = "/Users/woni/AI_TRADING/data/market.duckdb"

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("ALPHA DIAGNOSTIC")
print("=" * 70)


# ============================================================
# 1. Target distribution by period
# ============================================================

query = """
SELECT
    CASE
        WHEN date <= DATE '2024-12-31' THEN 'Train'
        WHEN date <= DATE '2025-12-31' THEN 'Validation'
        ELSE 'Test'
    END AS period,
    COUNT(*) AS rows,
    AVG(future_excess_return_5d) AS mean_target,
    STDDEV(future_excess_return_5d) AS std_target,
    MIN(future_excess_return_5d) AS min_target,
    MEDIAN(future_excess_return_5d) AS median_target,
    MAX(future_excess_return_5d) AS max_target
FROM stock_features
GROUP BY 1
ORDER BY 1
"""

df = con.execute(query).df()

print()
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 2. Target distribution by year
# ============================================================

query = """
SELECT
    YEAR(date) AS year,
    COUNT(*) AS rows,
    COUNT(DISTINCT symbol) AS symbols,
    AVG(future_excess_return_5d) AS mean_target,
    STDDEV(future_excess_return_5d) AS std_target,
    MEDIAN(future_excess_return_5d) AS median_target,
    MIN(future_excess_return_5d) AS min_target,
    MAX(future_excess_return_5d) AS max_target
FROM stock_features
GROUP BY YEAR(date)
ORDER BY year
"""

df = con.execute(query).df()

print()
print("=" * 70)
print("TARGET BY YEAR")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 3. Cross-sectional target dispersion by year
# ============================================================

query = """
WITH daily_dispersion AS (
    SELECT
        date,
        STDDEV(future_excess_return_5d) AS daily_std
    FROM stock_features
    GROUP BY date
)
SELECT
    YEAR(date) AS year,
    COUNT(*) AS trading_days,
    AVG(daily_std) AS avg_cross_sectional_std,
    MEDIAN(daily_std) AS median_cross_sectional_std,
    MAX(daily_std) AS max_cross_sectional_std
FROM daily_dispersion
GROUP BY YEAR(date)
ORDER BY year
"""

df = con.execute(query).df()

print()
print("=" * 70)
print("CROSS-SECTIONAL TARGET DISPERSION")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 4. Extreme target observations
# ============================================================

query = """
SELECT
    symbol,
    date,
    close,
    future_stock_return_5d,
    future_kospi_return_5d,
    future_excess_return_5d
FROM stock_features
ORDER BY ABS(future_excess_return_5d) DESC
LIMIT 30
"""

df = con.execute(query).df()

print()
print("=" * 70)
print("TOP 30 EXTREME TARGETS")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 5. Feature / target correlation - entire period
# ============================================================

features = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma120_ratio",
    "volatility_20d",
    "volatility_60d",
    "volume_ratio_20d",
    "rsi_14",
    "atr_14",
    "atr_ratio_14",
    "kospi_return_1d",
    "kospi_return_5d",
    "kospi_return_20d",
    "kospi_return_60d",
    "relative_strength_5d",
    "relative_strength_20d",
    "relative_strength_60d",
]

select_cols = ", ".join(
    [
        f"CORR({feature}, future_excess_return_5d) AS {feature}"
        for feature in features
    ]
)

query = f"""
SELECT
    {select_cols}
FROM stock_features
"""

df = con.execute(query).df().T.reset_index()

df.columns = ["feature", "correlation"]

df["abs_correlation"] = df["correlation"].abs()

df = df.sort_values(
    "abs_correlation",
    ascending=False
)

print()
print("=" * 70)
print("FEATURE / TARGET CORRELATION - ALL PERIOD")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 6. Feature / target correlation - Test period
# ============================================================

query = f"""
SELECT
    {select_cols}
FROM stock_features
WHERE date >= DATE '2026-01-01'
"""

df = con.execute(query).df().T.reset_index()

df.columns = ["feature", "correlation"]

df["abs_correlation"] = df["correlation"].abs()

df = df.sort_values(
    "abs_correlation",
    ascending=False
)

print()
print("=" * 70)
print("FEATURE / TARGET CORRELATION - TEST 2026")
print("=" * 70)
print(df.to_string(index=False))


# ============================================================
# 7. Feature / target correlation - Validation period
# ============================================================

query = f"""
SELECT
    {select_cols}
FROM stock_features
WHERE date >= DATE '2025-01-01'
  AND date <= DATE '2025-12-31'
"""

df = con.execute(query).df().T.reset_index()

df.columns = ["feature", "correlation"]

df["abs_correlation"] = df["correlation"].abs()

df = df.sort_values(
    "abs_correlation",
    ascending=False
)

print()
print("=" * 70)
print("FEATURE / TARGET CORRELATION - VALIDATION 2025")
print("=" * 70)
print(df.to_string(index=False))


print()
print("=" * 70)
print("DONE")
print("=" * 70)

con.close()