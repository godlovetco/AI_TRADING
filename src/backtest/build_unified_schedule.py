import duckdb
import pandas as pd

DB_PATH = "data/market.duckdb"

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-09-30")

HOLDING_DAYS = 5

OUTPUT = "data/risk_unified_rebalance_schedule.csv"

con = duckdb.connect(DB_PATH)

df = con.execute(
    """
    SELECT DISTINCT
        CAST(date AS DATE) AS date
    FROM alpha_lgbm_predictions
    WHERE date >= ?
      AND date <= ?
    ORDER BY date
    """,
    [START, END],
).fetchdf()

con.close()

df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------------------------
# Canonical global 5-prediction-date schedule
# ------------------------------------------------------------

schedule = df.iloc[::HOLDING_DAYS].copy()

schedule["rebalance_id"] = range(
    len(schedule)
)

schedule["year"] = (
    schedule["date"]
    .dt.year
)

schedule = schedule[
    [
        "rebalance_id",
        "date",
        "year",
    ]
]

schedule.to_csv(
    OUTPUT,
    index=False,
)

print()
print("=" * 72)
print("UNIFIED REBALANCE SCHEDULE")
print("=" * 72)

print(
    f"Prediction dates : {len(df):,}"
)

print(
    f"Rebalance dates  : {len(schedule):,}"
)

print()
print(
    schedule
    .groupby("year")
    .size()
    .to_string()
)

print()
print("First 10:")
print(
    schedule.head(10)
    .to_string(index=False)
)

print()
print("Last 10:")
print(
    schedule.tail(10)
    .to_string(index=False)
)

print()
print(f"Saved: {OUTPUT}")
