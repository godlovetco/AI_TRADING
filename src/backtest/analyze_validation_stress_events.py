from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DB = "data/market.duckdb"
SCHEDULE_FILE = "data/risk_validation_rebalance_schedule.csv"


# ============================================================
# Load validation schedule
# ============================================================
schedule = pd.read_csv(SCHEDULE_FILE)
schedule["date"] = pd.to_datetime(schedule["date"])

dates = schedule["date"].tolist()


# ============================================================
# Load KOSPI
# ============================================================
con = duckdb.connect(DB)

kospi = con.execute(
    """
    SELECT
        date,
        close
    FROM kospi_daily
    ORDER BY date
    """
).df()

kospi["date"] = pd.to_datetime(kospi["date"])
kospi = kospi.sort_values("date").reset_index(drop=True)

kospi["ret_5d_forward"] = (
    kospi["close"].shift(-5) / kospi["close"] - 1
)

kospi["ret_20d_forward"] = (
    kospi["close"].shift(-20) / kospi["close"] - 1
)


# ============================================================
# Load regime information
# ============================================================
prices = con.execute(
    """
    SELECT
        symbol,
        date,
        close
    FROM daily_prices
    ORDER BY symbol, date
    """
).df()

prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["symbol", "date"])

prices["ret_1d"] = prices.groupby("symbol")["close"].pct_change()

# Market cross-sectional return
market = (
    prices.groupby("date")["ret_1d"]
    .mean()
    .rename("market_ret")
    .reset_index()
)

market["ret_5d"] = (
    market["market_ret"]
    .rolling(5)
    .apply(lambda x: np.prod(1 + x) - 1)
)

market["ret_20d"] = (
    market["market_ret"]
    .rolling(20)
    .apply(lambda x: np.prod(1 + x) - 1)
)

market["vol_20d"] = (
    market["market_ret"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

# 60-day breadth:
# fraction of stocks whose 60-day return is positive
stock_ret60 = (
    prices.groupby("symbol")["close"]
    .pct_change(60)
)

breadth = (
    stock_ret60.groupby(prices["date"])
    .apply(lambda x: (x > 0).mean())
    .rename("breadth")
    .reset_index()
)

breadth["breadth_ma60"] = (
    breadth["breadth"]
    .rolling(60)
    .mean()
)

regime = market.merge(
    breadth,
    on="date",
    how="left"
)

regime = regime[
    regime["date"].isin(dates)
].copy()


# ============================================================
# Candidate stress conditions
# ============================================================
regime["stress_vol30_breadth40"] = (
    (regime["vol_20d"] >= 0.30)
    & (regime["breadth_ma60"] <= 0.40)
)

events = regime[
    regime["stress_vol30_breadth40"]
].copy()

events = events.sort_values("date")


# ============================================================
# Forward market performance
# ============================================================
events = events.merge(
    kospi[
        [
            "date",
            "close",
            "ret_5d_forward",
            "ret_20d_forward",
        ]
    ],
    on="date",
    how="left"
)

print()
print("=" * 100)
print("VALIDATION STRESS EVENTS")
print("=" * 100)

print(f"Stress events: {len(events)}")

if len(events):
    print()
    print(
        events[
            [
                "date",
                "ret_5d",
                "ret_20d",
                "vol_20d",
                "breadth_ma60",
                "ret_5d_forward",
                "ret_20d_forward",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("FORWARD MARKET PERFORMANCE")
    print("=" * 100)

    print(
        events[
            [
                "ret_5d_forward",
                "ret_20d_forward",
            ]
        ]
        .describe()
        .to_string()
    )


# ============================================================
# Also show all high-vol dates
# ============================================================
high_vol = regime[
    regime["vol_20d"] >= 0.30
].copy()

high_vol = high_vol.merge(
    kospi[
        [
            "date",
            "ret_5d_forward",
            "ret_20d_forward",
        ]
    ],
    on="date",
    how="left"
)

print()
print("=" * 100)
print("ALL HIGH-VOLATILITY EVENTS")
print("=" * 100)

print(f"High-vol events: {len(high_vol)}")

print(
    high_vol[
        [
            "date",
            "ret_20d",
            "vol_20d",
            "breadth_ma60",
            "ret_5d_forward",
            "ret_20d_forward",
        ]
    ]
    .sort_values("vol_20d", ascending=False)
    .to_string(index=False)
)

OUTPUT = "data/validation_stress_events.csv"
events.to_csv(OUTPUT, index=False)

print()
print(f"Saved: {OUTPUT}")
