import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = "data/market.duckdb"
OUTPUT_PATH = "data/alpha_regime_analysis.csv"

START_DATE = "2020-01-01"
END_DATE = "2026-09-22"

TOP_PCT = 0.20
BOTTOM_PCT = 0.20

conn = duckdb.connect(DB_PATH)

print("=" * 70)
print("ALPHA REGIME ANALYSIS")
print("=" * 70)

# ============================================================
# 1. Load predictions
# ============================================================

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

print(f"Prediction rows : {len(pred):,}")
print(f"Symbols         : {pred['symbol'].nunique()}")
print(f"Dates           : {pred['prediction_date'].nunique()}")

# ============================================================
# 2. Load KOSPI
# ============================================================

kospi = conn.execute("""
    SELECT
        date::DATE AS date,
        close
    FROM kospi_daily
    WHERE date >= ?
      AND date <= ?
    ORDER BY date
""", [START_DATE, END_DATE]).df()

kospi["date"] = pd.to_datetime(kospi["date"])
kospi = kospi.sort_values("date").reset_index(drop=True)

kospi["ret_1d"] = kospi["close"].pct_change(1)
kospi["ret_5d"] = kospi["close"].pct_change(5)
kospi["ret_20d"] = kospi["close"].pct_change(20)
kospi["ret_60d"] = kospi["close"].pct_change(60)

kospi["vol_5d"] = kospi["ret_1d"].rolling(5).std() * np.sqrt(252)
kospi["vol_20d"] = kospi["ret_1d"].rolling(20).std() * np.sqrt(252)
kospi["vol_60d"] = kospi["ret_1d"].rolling(60).std() * np.sqrt(252)

kospi["vol_ratio_20_60"] = kospi["vol_20d"] / kospi["vol_60d"]

kospi["ma20"] = kospi["close"].rolling(20).mean()
kospi["ma60"] = kospi["close"].rolling(60).mean()
kospi["ma120"] = kospi["close"].rolling(120).mean()

kospi["ma20_ratio"] = kospi["close"] / kospi["ma20"] - 1
kospi["ma60_ratio"] = kospi["close"] / kospi["ma60"] - 1
kospi["ma120_ratio"] = kospi["close"] / kospi["ma120"] - 1

# ============================================================
# 3. Load stock prices
# ============================================================

prices = conn.execute("""
    SELECT
        symbol,
        date::DATE AS date,
        open,
        close
    FROM daily_prices
    WHERE date >= ?
      AND date <= ?
    ORDER BY symbol, date
""", [START_DATE, END_DATE]).df()

prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["symbol", "date"])

prices["ret_1d"] = (
    prices.groupby("symbol")["close"]
    .pct_change()
)

# ============================================================
# 4. Market breadth
# ============================================================

breadth = (
    prices
    .merge(
        prices[["symbol", "date", "close"]]
        .assign(
            ma20=lambda x:
                x.groupby("symbol")["close"]
                .transform(lambda s: s.rolling(20).mean()),
            ma60=lambda x:
                x.groupby("symbol")["close"]
                .transform(lambda s: s.rolling(60).mean())
        ),
        on=["symbol", "date", "close"],
        how="left"
    )
)

breadth["above_ma20"] = (
    breadth["close"] > breadth["ma20"]
)

breadth["above_ma60"] = (
    breadth["close"] > breadth["ma60"]
)

breadth_daily = (
    breadth
    .groupby("date")
    .agg(
        breadth_ma20=("above_ma20", "mean"),
        breadth_ma60=("above_ma60", "mean"),
        market_up_ratio=("ret_1d", lambda x: (x > 0).mean()),
        market_down_ratio=("ret_1d", lambda x: (x < 0).mean())
    )
    .reset_index()
)

# ============================================================
# 5. Merge regime variables
# ============================================================

regime = kospi.merge(
    breadth_daily,
    on="date",
    how="left"
)

regime = regime.sort_values("date")

# ============================================================
# 6. Define regimes
# ============================================================

# Volatility regime
regime["vol_regime"] = np.select(
    [
        regime["vol_20d"] >= regime["vol_20d"].quantile(0.80),
        regime["vol_20d"] <= regime["vol_20d"].quantile(0.20)
    ],
    [
        "HIGH_VOL",
        "LOW_VOL"
    ],
    default="NORMAL_VOL"
)

# Volatility expansion
regime["vol_expansion"] = np.select(
    [
        regime["vol_ratio_20_60"] >= 1.30,
        regime["vol_ratio_20_60"] <= 0.80
    ],
    [
        "VOL_EXPANSION",
        "VOL_CONTRACTION"
    ],
    default="VOL_NORMAL"
)

# Market trend
regime["trend_regime"] = np.select(
    [
        (regime["close"] > regime["ma60"]) &
        (regime["ma60"] > regime["ma120"]),

        (regime["close"] < regime["ma60"]) &
        (regime["ma60"] < regime["ma120"])
    ],
    [
        "BULL",
        "BEAR"
    ],
    default="TRANSITION"
)

# Market momentum
regime["momentum_regime"] = np.select(
    [
        regime["ret_20d"] >= 0.10,
        regime["ret_20d"] <= -0.10
    ],
    [
        "STRONG_UP",
        "STRONG_DOWN"
    ],
    default="NEUTRAL"
)

# Breadth regime
regime["breadth_regime"] = np.select(
    [
        regime["breadth_ma60"] >= 0.70,
        regime["breadth_ma60"] <= 0.30
    ],
    [
        "BROAD_BULL",
        "BROAD_BEAR"
    ],
    default="MIXED"
)

# Stress regime
regime["stress_regime"] = np.select(
    [
        (
            (regime["vol_20d"] >= regime["vol_20d"].quantile(0.80))
            &
            (regime["ret_5d"] <= -0.05)
        ),

        (
            (regime["vol_20d"] >= regime["vol_20d"].quantile(0.80))
            &
            (regime["ret_5d"] >= 0.05)
        )
    ],
    [
        "DOWNSIDE_STRESS",
        "HIGH_VOL_RALLY"
    ],
    default="NORMAL"
)

# ============================================================
# 7. Merge prediction + regime
# ============================================================

pred = pred.merge(
    regime,
    left_on="prediction_date",
    right_on="date",
    how="left"
)

# ============================================================
# 8. Calculate realized 5-day return
# ============================================================

price_lookup = prices[
    ["symbol", "date", "open"]
].copy()

price_lookup = price_lookup.rename(
    columns={"date": "entry_date", "open": "entry_open"}
)

pred_dates = sorted(
    pred["prediction_date"].dropna().unique()
)

rows = []

for prediction_date in pred_dates:

    day_pred = pred[
        pred["prediction_date"] == prediction_date
    ].copy()

    if len(day_pred) < 20:
        continue

    future_dates = prices[
        prices["date"] > prediction_date
    ]["date"].drop_duplicates().sort_values().tolist()

    if len(future_dates) < 6:
        continue

    entry_date = future_dates[0]
    exit_date = future_dates[5]

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

    day_pred = (
        day_pred
        .merge(entry, on="symbol", how="left")
        .merge(exit_, on="symbol", how="left")
    )

    day_pred["realized_return"] = (
        day_pred["exit_open"] /
        day_pred["entry_open"] - 1
    )

    day_pred = day_pred.dropna(
        subset=["prediction", "realized_return"]
    )

    if len(day_pred) < 20:
        continue

    day_pred = day_pred.sort_values(
        "prediction"
    ).reset_index(drop=True)

    n = len(day_pred)

    top_n = max(1, int(n * TOP_PCT))
    bottom_n = max(1, int(n * BOTTOM_PCT))

    top = day_pred.tail(top_n)
    bottom = day_pred.head(bottom_n)

    # Rank IC
    rank_ic = (
        day_pred["prediction"]
        .rank()
        .corr(
            day_pred["realized_return"].rank()
        )
    )

    pearson = day_pred["prediction"].corr(
        day_pred["realized_return"]
    )

    top_return = top["realized_return"].mean()
    bottom_return = bottom["realized_return"].mean()

    row = {
        "prediction_date": prediction_date,

        "universe": n,

        "top_return": top_return,
        "bottom_return": bottom_return,
        "top_bottom_spread":
            top_return - bottom_return,

        "rank_ic": rank_ic,
        "pearson": pearson,

        "top_win_rate":
            (top["realized_return"] > 0).mean(),

        "bottom_win_rate":
            (bottom["realized_return"] > 0).mean(),
    }

    # Add regime values
    regime_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "vol_5d",
        "vol_20d",
        "vol_60d",
        "vol_ratio_20_60",
        "ma20_ratio",
        "ma60_ratio",
        "ma120_ratio",
        "breadth_ma20",
        "breadth_ma60",
        "market_up_ratio",
        "market_down_ratio",
        "vol_regime",
        "vol_expansion",
        "trend_regime",
        "momentum_regime",
        "breadth_regime",
        "stress_regime"
    ]

    r = day_pred.iloc[0]

    for col in regime_cols:
        row[col] = r.get(col, np.nan)

    rows.append(row)

result = pd.DataFrame(rows)

# ============================================================
# 9. Analysis helper
# ============================================================

def summarize(df, group_col):

    out = (
        df.groupby(group_col)
        .agg(
            periods=("prediction_date", "count"),

            mean_rank_ic=("rank_ic", "mean"),
            median_rank_ic=("rank_ic", "median"),

            ic_std=("rank_ic", "std"),

            positive_ic_ratio=(
                "rank_ic",
                lambda x: (x > 0).mean()
            ),

            mean_top_return=("top_return", "mean"),
            mean_bottom_return=("bottom_return", "mean"),
            mean_spread=("top_bottom_spread", "mean"),

            spread_positive_ratio=(
                "top_bottom_spread",
                lambda x: (x > 0).mean()
            ),

            mean_top_win_rate=("top_win_rate", "mean")
        )
        .reset_index()
    )

    out["ic_ir"] = (
        out["mean_rank_ic"] /
        out["ic_std"]
        * np.sqrt(252)
    )

    return out


# ============================================================
# 10. Print results
# ============================================================

print()
print("=" * 70)
print("OVERALL")
print("=" * 70)

overall = {
    "periods": len(result),
    "mean_rank_ic": result["rank_ic"].mean(),
    "ic_ir": (
        result["rank_ic"].mean() /
        result["rank_ic"].std()
        * np.sqrt(252)
    ),
    "positive_ic_ratio":
        (result["rank_ic"] > 0).mean(),
    "mean_top_return":
        result["top_return"].mean(),
    "mean_bottom_return":
        result["bottom_return"].mean(),
    "mean_spread":
        result["top_bottom_spread"].mean(),
    "spread_positive_ratio":
        (result["top_bottom_spread"] > 0).mean()
}

for k, v in overall.items():
    if isinstance(v, float):
        print(f"{k:25s}: {v:.6f}")
    else:
        print(f"{k:25s}: {v}")


for col in [
    "vol_regime",
    "vol_expansion",
    "trend_regime",
    "momentum_regime",
    "breadth_regime",
    "stress_regime"
]:

    print()
    print("=" * 70)
    print(col.upper())
    print("=" * 70)

    summary = summarize(result, col)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 11. Find worst Alpha regimes
# ============================================================

print()
print("=" * 70)
print("WORST 20 ALPHA PERIODS")
print("=" * 70)

worst = (
    result
    .sort_values("top_bottom_spread")
    .head(20)
)

print(
    worst[
        [
            "prediction_date",
            "rank_ic",
            "top_bottom_spread",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "vol_ratio_20_60",
            "breadth_ma60",
            "stress_regime"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 12. Save
# ============================================================

Path("data").mkdir(exist_ok=True)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)
print(OUTPUT_PATH)

conn.close()
