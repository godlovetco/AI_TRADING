import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/market.duckdb"
SCHEDULE_FILE = "data/risk_validation_rebalance_schedule.csv"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

con = duckdb.connect(DB_PATH)

# ------------------------------------------------------------
# Load prices
# ------------------------------------------------------------

prices = con.execute(
    """
    SELECT
        CAST(date AS DATE) AS date,
        symbol,
        close
    FROM daily_prices
    WHERE date <= ?
    ORDER BY symbol, date
    """,
    [END],
).fetchdf()

con.close()

prices["date"] = pd.to_datetime(prices["date"])

prices = prices.sort_values(
    ["symbol", "date"]
)

prices["ret_1d"] = (
    prices
    .groupby("symbol")["close"]
    .pct_change()
)

# ------------------------------------------------------------
# Market return
# ------------------------------------------------------------

market = (
    prices
    .groupby("date")["ret_1d"]
    .mean()
    .rename("market_ret")
    .reset_index()
)

# Use compounded returns rather than summing daily returns.
market["ret_5d"] = (
    (1.0 + market["market_ret"])
    .rolling(5)
    .apply(np.prod, raw=True)
    - 1.0
)

market["ret_20d"] = (
    (1.0 + market["market_ret"])
    .rolling(20)
    .apply(np.prod, raw=True)
    - 1.0
)

market["vol_20d"] = (
    market["market_ret"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

# ------------------------------------------------------------
# Breadth
# ------------------------------------------------------------

stock = prices.copy()

stock["ret_60d"] = (
    stock
    .groupby("symbol")["close"]
    .pct_change(60)
)

daily_breadth = (
    stock
    .groupby("date")["ret_60d"]
    .apply(
        lambda x: (x > 0).mean()
    )
    .rename("breadth")
    .reset_index()
)

# 60-day moving average breadth
daily_breadth["breadth_ma60"] = (
    daily_breadth["breadth"]
    .rolling(60)
    .mean()
)

regime = market.merge(
    daily_breadth[
        [
            "date",
            "breadth",
            "breadth_ma60",
        ]
    ],
    on="date",
    how="left",
)

# ------------------------------------------------------------
# Unified Schedule
# ------------------------------------------------------------

schedule = pd.read_csv(
    SCHEDULE_FILE
)

schedule["date"] = pd.to_datetime(
    schedule["date"]
)

schedule = schedule[
    (schedule["date"] >= START)
    & (schedule["date"] <= END)
].copy()

# Last schedule point is not used because there is no complete
# 5-day holding period.
schedule = schedule.iloc[:-1].copy()

print()
print("=" * 80)
print("V4 REGIME THRESHOLD SCAN")
print("=" * 80)

print(
    f"Unified schedule dates : {len(schedule)}"
)

# ------------------------------------------------------------
# Merge regime into schedule
# ------------------------------------------------------------

schedule = schedule.merge(
    regime,
    on="date",
    how="left",
)

print(
    f"Regime valid dates     : "
    f"{schedule['vol_20d'].notna().sum()}"
)

print()
print("REGIME DIAGNOSTICS")
print(
    schedule[
        [
            "date",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("REGIME NaN COUNT")
print(
    schedule[
        [
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
        ]
    ]
    .isna()
    .sum()
    .to_string()
)

# ------------------------------------------------------------
# Candidate conditions
# ------------------------------------------------------------

def v3_current(x):

    return (
        (
            (x["vol_20d"] >= 0.40)
            & (x["breadth_ma60"] <= 0.30)
        )
        |
        (x["ret_20d"] <= -0.15)
        |
        (
            (x["ret_20d"] >= 0.10)
            & (x["ret_5d"] <= -0.05)
        )
    )


def strict_20_breadth40(x):

    return (
        (
            (x["vol_20d"] >= 0.40)
            & (x["breadth_ma60"] <= 0.30)
        )
        |
        (
            (x["ret_20d"] <= -0.20)
            & (x["breadth_ma60"] <= 0.40)
        )
        |
        (
            (x["ret_20d"] >= 0.10)
            & (x["ret_5d"] <= -0.05)
            & (x["vol_20d"] >= 0.40)
        )
    )


def strict_20_breadth30(x):

    return (
        (
            (x["vol_20d"] >= 0.40)
            & (x["breadth_ma60"] <= 0.30)
        )
        |
        (
            (x["ret_20d"] <= -0.20)
            & (x["breadth_ma60"] <= 0.30)
        )
        |
        (
            (x["ret_20d"] >= 0.10)
            & (x["ret_5d"] <= -0.05)
            & (x["vol_20d"] >= 0.40)
        )
    )


def strict_25_breadth40(x):

    return (
        (
            (x["vol_20d"] >= 0.40)
            & (x["breadth_ma60"] <= 0.30)
        )
        |
        (
            (x["ret_20d"] <= -0.25)
            & (x["breadth_ma60"] <= 0.40)
        )
        |
        (
            (x["ret_20d"] >= 0.10)
            & (x["ret_5d"] <= -0.05)
            & (x["vol_20d"] >= 0.40)
        )
    )


conditions = {
    "V3_CURRENT": v3_current,
    "STRICT_20_BREADTH40": strict_20_breadth40,
    "STRICT_20_BREADTH30": strict_20_breadth30,
    "STRICT_25_BREADTH40": strict_25_breadth40,
}

# ------------------------------------------------------------
# Evaluate conditions
# ------------------------------------------------------------

results = []

for name, func in conditions.items():

    temp = schedule.copy()

    temp["severe"] = (
        temp.apply(
            func,
            axis=1,
        )
    )

    # HVR condition
    temp["high_vol_rally"] = (
        (temp["vol_20d"] >= 0.40)
        &
        (temp["ret_20d"] >= 0.10)
    )

    # Two-stage exposure
    temp["exposure"] = 1.0

    temp.loc[
        temp["high_vol_rally"],
        "exposure"
    ] = 0.50

    temp.loc[
        temp["severe"],
        "exposure"
    ] = 0.25

    # Important:
    # if severe is also HVR, severe takes precedence.

    severe_returns = []

    for _, row in temp.iterrows():

        # The scan here focuses on the regime itself.
        # Use market return as the regime shock proxy.
        severe_returns.append(
            row["ret_20d"]
        )

    temp["market_proxy_return"] = (
        severe_returns
    )

    results.append(
        {
            "condition": name,
            "periods": len(temp),
            "severe_periods": int(
                temp["severe"].sum()
            ),
            "severe_ratio": (
                temp["severe"].mean()
            ),
            "hvr_periods": int(
                temp["high_vol_rally"].sum()
            ),
            "mean_exposure": (
                temp["exposure"].mean()
            ),
            "min_exposure": (
                temp["exposure"].min()
            ),
            "mean_severe_ret20": (
                temp.loc[
                    temp["severe"],
                    "ret_20d"
                ].mean()
                if temp["severe"].any()
                else np.nan
            ),
        }
    )

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        temp.loc[
            temp["severe"],
            [
                "date",
                "ret_5d",
                "ret_20d",
                "vol_20d",
                "breadth_ma60",
                "high_vol_rally",
                "exposure",
            ]
        ]
        .to_string(index=False)
    )

result_df = pd.DataFrame(
    results
)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

output = (
    "data/v4_severe_stress_scan_2025.csv"
)

result_df.to_csv(
    output,
    index=False,
)

print()
print(f"Saved: {output}")
