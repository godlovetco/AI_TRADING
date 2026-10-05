from pathlib import Path

import pandas as pd


START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

SCHEDULE_FILE = "data/risk_validation_rebalance_schedule.csv"
OUTPUT = "data/validation_regime_distribution.csv"


# ---------------------------------------------------------------------
# Load schedule
# ---------------------------------------------------------------------
schedule = pd.read_csv(SCHEDULE_FILE)
schedule["date"] = pd.to_datetime(schedule["date"])

schedule = schedule[
    (schedule["date"] >= START)
    & (schedule["date"] <= END)
].copy()

schedule_dates = set(schedule["date"])


# ---------------------------------------------------------------------
# Load regime data from the existing scan script
# ---------------------------------------------------------------------
# Reproduce the same calculations used by scan_v4_severe_stress.py.
#
# We import the module so the regime construction remains identical.
# ---------------------------------------------------------------------
import sys

sys.path.insert(0, "src/backtest")

import scan_v4_severe_stress as scan


# The scan script should already contain the regime dataframe.
# Try common variable names.
regime = None

for name in ["regime", "regime_df", "df_regime", "features"]:
    if hasattr(scan, name):
        candidate = getattr(scan, name)
        if isinstance(candidate, pd.DataFrame):
            if "date" in candidate.columns:
                regime = candidate.copy()
                break

if regime is None:
    raise RuntimeError(
        "Could not find regime DataFrame in scan_v4_severe_stress.py"
    )


regime["date"] = pd.to_datetime(regime["date"])

df = regime[
    regime["date"].isin(schedule_dates)
].copy()

required = [
    "date",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "breadth_ma60",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing columns: {missing}")


# ---------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------
print()
print("=" * 80)
print("VALIDATION REGIME DISTRIBUTION")
print("=" * 80)

print(f"Periods: {len(df)}")

for col in [
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "breadth_ma60",
]:
    print()
    print("-" * 80)
    print(col)

    print(
        df[col]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )


# ---------------------------------------------------------------------
# Extreme dates
# ---------------------------------------------------------------------
print()
print("=" * 80)
print("EXTREME RET_20D DOWNSIDE")
print("=" * 80)

print(
    df.sort_values("ret_20d")
    .head(20)
    [
        [
            "date",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
        ]
    ]
    .to_string(index=False)
)


print()
print("=" * 80)
print("EXTREME VOLATILITY")
print("=" * 80)

print(
    df.sort_values("vol_20d", ascending=False)
    .head(20)
    [
        [
            "date",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
        ]
    ]
    .to_string(index=False)
)


print()
print("=" * 80)
print("EXTREME BEARISH BREADTH")
print("=" * 80)

print(
    df.sort_values("breadth_ma60")
    .head(20)
    [
        [
            "date",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
        ]
    ]
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Candidate trigger counts
# ---------------------------------------------------------------------
conditions = {
    "VOL_30": df["vol_20d"] >= 0.30,
    "VOL_35": df["vol_20d"] >= 0.35,
    "VOL_40": df["vol_20d"] >= 0.40,

    "DOWN_10": df["ret_20d"] <= -0.10,
    "DOWN_15": df["ret_20d"] <= -0.15,
    "DOWN_20": df["ret_20d"] <= -0.20,

    "BREADTH_30": df["breadth_ma60"] <= 0.30,
    "BREADTH_35": df["breadth_ma60"] <= 0.35,
    "BREADTH_40": df["breadth_ma60"] <= 0.40,

    "VOL30_BREADTH40": (
        (df["vol_20d"] >= 0.30)
        & (df["breadth_ma60"] <= 0.40)
    ),

    "VOL35_BREADTH40": (
        (df["vol_20d"] >= 0.35)
        & (df["breadth_ma60"] <= 0.40)
    ),

    "VOL30_BREADTH35": (
        (df["vol_20d"] >= 0.30)
        & (df["breadth_ma60"] <= 0.35)
    ),

    "DOWN15_BREADTH40": (
        (df["ret_20d"] <= -0.15)
        & (df["breadth_ma60"] <= 0.40)
    ),

    "DOWN20_BREADTH40": (
        (df["ret_20d"] <= -0.20)
        & (df["breadth_ma60"] <= 0.40)
    ),
}


print()
print("=" * 80)
print("CANDIDATE TRIGGER FREQUENCY")
print("=" * 80)

rows = []

for name, mask in conditions.items():
    count = int(mask.sum())

    rows.append(
        {
            "condition": name,
            "periods": len(df),
            "trigger_count": count,
            "trigger_ratio": count / len(df) if len(df) else 0,
        }
    )

summary = pd.DataFrame(rows)

print(summary.to_string(index=False))

summary.to_csv(OUTPUT, index=False)

print()
print(f"Saved: {OUTPUT}")
