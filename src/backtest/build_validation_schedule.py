import duckdb
import pandas as pd

DB_PATH = "data/market.duckdb"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

OUTPUT = "data/risk_validation_rebalance_schedule.csv"

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

schedule = df.iloc[::5].copy()

schedule["rebalance_id"] = range(
    len(schedule)
)

schedule["year"] = (
    schedule["date"].dt.year
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
print("=" * 80)
print("VALIDATION REBALANCE SCHEDULE")
print("=" * 80)

print(
    f"Prediction dates : {len(df)}"
)

print(
    f"Rebalance dates  : {len(schedule)}"
)

print()
print(
    schedule
    .groupby("year")
    .size()
    .to_string()
)

print()
print("First 5:")
print(
    schedule.head()
    .to_string(index=False)
)

print()
print("Last 5:")
print(
    schedule.tail()
    .to_string(index=False)
)

print()
print(f"Saved: {OUTPUT}")
