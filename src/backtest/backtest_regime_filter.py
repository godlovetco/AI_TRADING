import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = "data/market.duckdb"
REGIME_PATH = "data/alpha_regime_analysis.csv"

START_DATE = "2025-01-01"
END_DATE = "2026-09-22"

TOP_PCT = 0.20
TRANSACTION_COST = 0.001

HOLD_DAYS = 5


# ============================================================
# Load
# ============================================================

conn = duckdb.connect(DB_PATH)

pred = conn.execute("""
    SELECT
        symbol,
        date::DATE AS prediction_date,
        prediction
    FROM alpha_lgbm_predictions
    WHERE date >= ?
      AND date <= ?
    ORDER BY date, symbol
""", [START_DATE, END_DATE]).df()

pred["prediction_date"] = pd.to_datetime(
    pred["prediction_date"]
)

prices = conn.execute("""
    SELECT
        symbol,
        date::DATE AS date,
        open
    FROM daily_prices
    WHERE date >= ?
      AND date <= ?
    ORDER BY symbol, date
""", [START_DATE, END_DATE]).df()

prices["date"] = pd.to_datetime(prices["date"])

regime = pd.read_csv(REGIME_PATH)

regime["prediction_date"] = pd.to_datetime(
    regime["prediction_date"]
)

conn.close()


# ============================================================
# Build trading periods
# ============================================================

prediction_dates = sorted(
    pred["prediction_date"].unique()
)

# Need enough future prices
price_dates = sorted(
    prices["date"].unique()
)

price_date_set = set(price_dates)

periods = []

i = 0

while i < len(prediction_dates):

    prediction_date = prediction_dates[i]

    future_dates = [
        d for d in price_dates
        if d > prediction_date
    ]

    if len(future_dates) < HOLD_DAYS + 1:
        break

    entry_date = future_dates[0]
    exit_date = future_dates[HOLD_DAYS]

    periods.append(
        {
            "prediction_date": prediction_date,
            "entry_date": entry_date,
            "exit_date": exit_date
        }
    )

    # Non-overlapping 5 trading day periods
    i += HOLD_DAYS


periods = pd.DataFrame(periods)

print("=" * 70)
print("REGIME FILTER BACKTEST")
print("=" * 70)

print(f"Prediction dates : {len(prediction_dates)}")
print(f"Backtest periods : {len(periods)}")

print()


# ============================================================
# Calculate Top20 EW return for each period
# ============================================================

trades = []

for _, p in periods.iterrows():

    prediction_date = p["prediction_date"]
    entry_date = p["entry_date"]
    exit_date = p["exit_date"]

    day_pred = pred[
        pred["prediction_date"] == prediction_date
    ].copy()

    if len(day_pred) < 20:
        continue

    day_pred = day_pred.sort_values(
        "prediction",
        ascending=False
    )

    n_top = max(
        1,
        int(len(day_pred) * TOP_PCT)
    )

    top = day_pred.head(n_top)

    entry = prices[
        prices["date"] == entry_date
    ][["symbol", "open"]].rename(
        columns={"open": "entry_open"}
    )

    exit_ = prices[
        prices["date"] == exit_date
    ][["symbol", "open"]].rename(
        columns={"open": "exit_open"}
    )

    top = (
        top
        .merge(entry, on="symbol", how="left")
        .merge(exit_, on="symbol", how="left")
    )

    top = top.dropna(
        subset=["entry_open", "exit_open"]
    )

    if len(top) < max(5, n_top * 0.8):
        continue

    top["stock_return"] = (
        top["exit_open"] /
        top["entry_open"] - 1
    )

    gross_return = top["stock_return"].mean()

    # Regime data
    r = regime[
        regime["prediction_date"] == prediction_date
    ]

    if len(r) == 0:
        continue

    r = r.iloc[0]

    trades.append(
        {
            "prediction_date": prediction_date,
            "entry_date": entry_date,
            "exit_date": exit_date,

            "gross_return": gross_return,

            "vol_regime": r["vol_regime"],
            "vol_expansion": r["vol_expansion"],
            "trend_regime": r["trend_regime"],
            "momentum_regime": r["momentum_regime"],
            "breadth_regime": r["breadth_regime"],
            "stress_regime": r["stress_regime"],

            "ret_5d": r["ret_5d"],
            "ret_20d": r["ret_20d"],
            "vol_20d": r["vol_20d"],
            "vol_ratio_20_60": r["vol_ratio_20_60"],
            "breadth_ma60": r["breadth_ma60"],
            "breadth_ma20": r["breadth_ma20"],
        }
    )


trades = pd.DataFrame(trades)

print(f"Usable periods   : {len(trades)}")


# ============================================================
# Define filters
# ============================================================

def exposure_baseline(row):
    return 1.00


def exposure_high_vol_rally(row):
    if row["stress_regime"] == "HIGH_VOL_RALLY":
        return 0.50
    return 1.00


def exposure_high_vol(row):
    if row["vol_regime"] == "HIGH_VOL":
        return 0.50
    return 1.00


def exposure_breadth_stress(row):
    if row["breadth_ma60"] <= 0.30:
        return 0.50
    return 1.00


def exposure_severe_bear(row):
    if (
        (row["ret_20d"] <= -0.10)
        or
        (row["breadth_ma60"] <= 0.30)
    ):
        return 0.50
    return 1.00


def exposure_reversal(row):
    # Strong recent rally followed by sharp short-term reversal
    if (
        (row["ret_20d"] >= 0.10)
        and
        (row["ret_5d"] <= -0.05)
    ):
        return 0.50
    return 1.00


def exposure_composite_stress(row):

    extreme_vol = (
        row["vol_20d"] >= 0.40
    )

    weak_breadth = (
        row["breadth_ma60"] <= 0.30
    )

    severe_downtrend = (
        row["ret_20d"] <= -0.10
    )

    reversal = (
        (row["ret_20d"] >= 0.10)
        and
        (row["ret_5d"] <= -0.05)
    )

    high_vol_rally = (
        row["stress_regime"] == "HIGH_VOL_RALLY"
    )

    if (
        (extreme_vol and weak_breadth)
        or severe_downtrend
        or reversal
        or high_vol_rally
    ):
        return 0.25

    return 1.00


FILTERS = {
    "BASELINE": exposure_baseline,
    "HIGH_VOL_RALLY_50": exposure_high_vol_rally,
    "HIGH_VOL_50": exposure_high_vol,
    "BREADTH_STRESS_50": exposure_breadth_stress,
    "SEVERE_BEAR_50": exposure_severe_bear,
    "REVERSAL_50": exposure_reversal,
    "COMPOSITE_STRESS_25": exposure_composite_stress,
}


# ============================================================
# Backtest engine
# ============================================================

def run_strategy(df, exposure_function):

    df = df.copy()

    df["exposure"] = df.apply(
        exposure_function,
        axis=1
    )

    # Portfolio return
    df["gross_portfolio_return"] = (
        df["gross_return"] *
        df["exposure"]
    )

    # --------------------------------------------------------
    # Turnover
    #
    # Exposure change:
    # previous exposure -> current exposure
    #
    # First entry:
    # full portfolio investment
    # --------------------------------------------------------

    previous_exposure = 0.0

    turnovers = []
    costs = []

    for exposure in df["exposure"]:

        turnover = abs(
            exposure - previous_exposure
        )

        turnovers.append(turnover)
        costs.append(
            turnover * TRANSACTION_COST
        )

        previous_exposure = exposure

    df["turnover"] = turnovers
    df["cost"] = costs

    df["net_return"] = (
        df["gross_portfolio_return"]
        - df["cost"]
    )

    # Equity curve
    df["equity"] = (
        1.0 + df["net_return"]
    ).cumprod()

    # Drawdown
    running_max = df["equity"].cummax()

    df["drawdown"] = (
        df["equity"] /
        running_max - 1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    n = len(df)

    total_return = (
        df["equity"].iloc[-1] - 1
    )

    years = n * HOLD_DAYS / 252

    annual_return = (
        (1 + total_return) ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    annual_volatility = (
        df["net_return"].std(ddof=1)
        * np.sqrt(252 / HOLD_DAYS)
    )

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    max_drawdown = df["drawdown"].min()

    win_rate = (
        df["net_return"] > 0
    ).mean()

    return {
        "periods": n,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "mean_gross": df["gross_portfolio_return"].mean(),
        "mean_net": df["net_return"].mean(),
        "mean_turnover": df["turnover"].mean(),
        "mean_exposure": df["exposure"].mean(),
        "min_exposure": df["exposure"].min(),
        "max_exposure": df["exposure"].max(),
        "exposure_reduced_ratio": (
            df["exposure"] < 1.0
        ).mean()
    }, df


# ============================================================
# Run all strategies
# ============================================================

results = []
detail = []

for name, func in FILTERS.items():

    metrics, path = run_strategy(
        trades,
        func
    )

    metrics["strategy"] = name

    results.append(metrics)

    path["strategy"] = name

    detail.append(path)


results = pd.DataFrame(results)

detail = pd.concat(
    detail,
    ignore_index=True
)


# ============================================================
# Print full results
# ============================================================

print()
print("=" * 70)
print("FULL PERIOD RESULTS")
print("=" * 70)

print(
    results[
        [
            "strategy",
            "periods",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "mean_net",
            "mean_exposure",
            "exposure_reduced_ratio"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 2025 / 2026 split
# ============================================================

detail["year"] = (
    pd.to_datetime(
        detail["prediction_date"]
    ).dt.year
)

year_results = []

for strategy in FILTERS:

    for year in [2025, 2026]:

        subset = detail[
            (detail["strategy"] == strategy)
            &
            (detail["year"] == year)
        ].copy()

        if len(subset) == 0:
            continue

        n = len(subset)

        total_return = (
            (1 + subset["net_return"]).prod() - 1
        )

        years = n * HOLD_DAYS / 252

        annual_return = (
            (1 + total_return) ** (1 / years) - 1
        )

        annual_volatility = (
            subset["net_return"].std(ddof=1)
            * np.sqrt(252 / HOLD_DAYS)
        )

        sharpe = (
            annual_return / annual_volatility
            if annual_volatility > 0
            else np.nan
        )

        equity = (
            1 + subset["net_return"]
        ).cumprod()

        drawdown = (
            equity /
            equity.cummax() - 1
        )

        year_results.append(
            {
                "strategy": strategy,
                "year": year,
                "periods": n,
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe": sharpe,
                "max_drawdown": drawdown.min(),
                "win_rate": (
                    subset["net_return"] > 0
                ).mean(),
                "mean_net": subset["net_return"].mean(),
                "mean_exposure": subset["exposure"].mean(),
                "reduced_ratio": (
                    subset["exposure"] < 1
                ).mean()
            }
        )


year_results = pd.DataFrame(
    year_results
)

print()
print("=" * 70)
print("2025 VALIDATION")
print("=" * 70)

print(
    year_results[
        year_results["year"] == 2025
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("=" * 70)
print("2026 TEST")
print("=" * 70)

print(
    year_results[
        year_results["year"] == 2026
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 2026 critical-event analysis
# ============================================================

print()
print("=" * 70)
print("2026 WORST PERIODS - BASELINE")
print("=" * 70)

base_2026 = detail[
    (detail["strategy"] == "BASELINE")
    &
    (detail["year"] == 2026)
].copy()

print(
    base_2026.sort_values(
        "net_return"
    ).head(15)[
        [
            "prediction_date",
            "entry_date",
            "exit_date",
            "gross_return",
            "net_return",
            "exposure",
            "vol_regime",
            "stress_regime",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "breadth_ma60"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# Save
# ============================================================

Path("data").mkdir(
    exist_ok=True
)

results.to_csv(
    "data/regime_filter_results.csv",
    index=False
)

year_results.to_csv(
    "data/regime_filter_year_results.csv",
    index=False
)

detail.to_csv(
    "data/regime_filter_detail.csv",
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print("data/regime_filter_results.csv")
print("data/regime_filter_year_results.csv")
print("data/regime_filter_detail.csv")
