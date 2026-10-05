import duckdb
import numpy as np
import pandas as pd


DB = "data/market.duckdb"
SCHEDULE_FILE = "data/risk_validation_rebalance_schedule.csv"


con = duckdb.connect(DB)

schedule = pd.read_csv(SCHEDULE_FILE)
schedule["date"] = pd.to_datetime(schedule["date"])

dates = set(schedule["date"])


# ------------------------------------------------------------
# Market data
# ------------------------------------------------------------
prices = con.execute(
    """
    SELECT symbol, date, close
    FROM daily_prices
    ORDER BY symbol, date
    """
).df()

prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["symbol", "date"])

prices["ret_1d"] = prices.groupby("symbol")["close"].pct_change()

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

market = market[market["date"].isin(dates)].copy()


# ------------------------------------------------------------
# Forward market returns
# ------------------------------------------------------------
kospi = con.execute(
    """
    SELECT date, close
    FROM kospi_daily
    ORDER BY date
    """
).df()

kospi["date"] = pd.to_datetime(kospi["date"])

kospi["forward_5d"] = (
    kospi["close"].shift(-5) / kospi["close"] - 1
)

kospi["forward_20d"] = (
    kospi["close"].shift(-20) / kospi["close"] - 1
)


df = market.merge(
    kospi[
        ["date", "forward_5d", "forward_20d"]
    ],
    on="date",
    how="left"
)


# ------------------------------------------------------------
# Candidate HVR definitions
# ------------------------------------------------------------
conditions = {
    "HVR_40_10_5": (
        (df["vol_20d"] >= 0.40)
        & (df["ret_20d"] >= 0.10)
        & (df["ret_5d"] <= -0.05)
    ),

    "HVR_35_10_5": (
        (df["vol_20d"] >= 0.35)
        & (df["ret_20d"] >= 0.10)
        & (df["ret_5d"] <= -0.05)
    ),

    "HVR_30_10_5": (
        (df["vol_20d"] >= 0.30)
        & (df["ret_20d"] >= 0.10)
        & (df["ret_5d"] <= -0.05)
    ),

    "HVR_40_08_5": (
        (df["vol_20d"] >= 0.40)
        & (df["ret_20d"] >= 0.08)
        & (df["ret_5d"] <= -0.05)
    ),
}


print()
print("=" * 100)
print("VALIDATION HVR EVENT ANALYSIS")
print("=" * 100)


rows = []

for name, mask in conditions.items():

    events = df[mask].copy()

    print()
    print("-" * 100)
    print(name)
    print(f"events: {len(events)}")

    if len(events):

        print(
            events[
                [
                    "date",
                    "ret_5d",
                    "ret_20d",
                    "vol_20d",
                    "forward_5d",
                    "forward_20d",
                ]
            ]
            .sort_values("date")
            .to_string(index=False)
        )

        rows.append(
            {
                "condition": name,
                "events": len(events),
                "mean_forward_5d":
                    events["forward_5d"].mean(),
                "mean_forward_20d":
                    events["forward_20d"].mean(),
                "positive_forward_5d":
                    (events["forward_5d"] > 0).mean(),
                "positive_forward_20d":
                    (events["forward_20d"] > 0).mean(),
            }
        )

    else:

        rows.append(
            {
                "condition": name,
                "events": 0,
                "mean_forward_5d": np.nan,
                "mean_forward_20d": np.nan,
                "positive_forward_5d": np.nan,
                "positive_forward_20d": np.nan,
            }
        )


summary = pd.DataFrame(rows)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(summary.to_string(index=False))

OUTPUT = "data/validation_hvr_events.csv"
summary.to_csv(OUTPUT, index=False)

print()
print(f"Saved: {OUTPUT}")
