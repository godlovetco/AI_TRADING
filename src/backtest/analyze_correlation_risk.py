import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/market.duckdb"

TOP_PCT = 0.20
CORR_LOOKBACK = 60

START_DATE = "2026-01-01"
END_DATE = "2026-09-22"

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("Correlation / Concentration Risk Analysis")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load predictions
# ------------------------------------------------------------

pred = con.sql(f"""
    SELECT
        date AS prediction_date,
        symbol,
        prediction
    FROM alpha_lgbm_predictions
    WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND prediction IS NOT NULL
    ORDER BY date, prediction DESC
""").df()

pred["prediction_date"] = pd.to_datetime(pred["prediction_date"])

print(f"Prediction rows : {len(pred):,}")
print(f"Prediction dates: {pred['prediction_date'].nunique():,}")
print(f"Symbols         : {pred['symbol'].nunique():,}")


# ------------------------------------------------------------
# 2. Load price history
# ------------------------------------------------------------

prices = con.sql(f"""
    SELECT
        date,
        symbol,
        close
    FROM daily_prices
    WHERE date <= '{END_DATE}'
    ORDER BY symbol, date
""").df()

prices["date"] = pd.to_datetime(prices["date"])

prices["ret_1d"] = (
    prices
    .groupby("symbol")["close"]
    .pct_change()
)


# ------------------------------------------------------------
# 3. Calculate correlation risk for each prediction date
# ------------------------------------------------------------

results = []

prediction_dates = sorted(
    pred["prediction_date"].unique()
)

for prediction_date in prediction_dates:

    day_pred = pred[
        pred["prediction_date"] == prediction_date
    ].copy()

    if len(day_pred) == 0:
        continue

    # Top 20%
    n_selected = max(
        1,
        int(np.floor(len(day_pred) * TOP_PCT))
    )

    selected = (
        day_pred
        .sort_values("prediction", ascending=False)
        .head(n_selected)
    )

    symbols = selected["symbol"].tolist()

    if len(symbols) < 2:
        continue

    # --------------------------------------------------------
    # Historical returns BEFORE prediction date
    # --------------------------------------------------------

    hist = prices[
        (prices["date"] < prediction_date)
        & (prices["symbol"].isin(symbols))
    ].copy()

    pivot = hist.pivot(
        index="date",
        columns="symbol",
        values="ret_1d"
    )

    # Require at least 80% of lookback observations
    valid_symbols = pivot.count()[
        pivot.count() >= CORR_LOOKBACK * 0.8
    ].index.tolist()

    pivot = pivot[valid_symbols]

    if len(valid_symbols) < 2:
        continue

    # Most recent 60 trading days
    pivot = pivot.tail(CORR_LOOKBACK)

    if len(pivot) < CORR_LOOKBACK * 0.8:
        continue

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    corr = pivot.corr()

    corr_values = corr.values[
        np.triu_indices_from(
            corr.values,
            k=1
        )
    ]

    corr_values = corr_values[
        np.isfinite(corr_values)
    ]

    if len(corr_values) == 0:
        continue

    mean_corr = float(
        np.mean(corr_values)
    )

    median_corr = float(
        np.median(corr_values)
    )

    p90_corr = float(
        np.quantile(corr_values, 0.90)
    )

    max_corr = float(
        np.max(corr_values)
    )

    high_corr_50 = float(
        np.mean(corr_values >= 0.50)
    )

    high_corr_70 = float(
        np.mean(corr_values >= 0.70)
    )

    high_corr_80 = float(
        np.mean(corr_values >= 0.80)
    )

    # --------------------------------------------------------
    # Correlation-adjusted Effective N
    # --------------------------------------------------------

    n = len(valid_symbols)

    denominator = (
        1
        + (n - 1) * mean_corr
    )

    if denominator > 0:
        effective_n_corr = (
            n / denominator
        )
    else:
        effective_n_corr = float(n)

    results.append({
        "prediction_date": prediction_date,
        "n_selected": len(symbols),
        "n_corr_symbols": n,
        "mean_corr": mean_corr,
        "median_corr": median_corr,
        "p90_corr": p90_corr,
        "max_corr": max_corr,
        "high_corr_50_pct": high_corr_50,
        "high_corr_70_pct": high_corr_70,
        "high_corr_80_pct": high_corr_80,
        "effective_n_corr": effective_n_corr,
    })


# ------------------------------------------------------------
# 4. Result
# ------------------------------------------------------------

result = pd.DataFrame(results)

print()
print("=" * 70)
print("2026 CORRELATION RISK SUMMARY")
print("=" * 70)

if len(result) == 0:

    print("No valid results.")
    con.close()
    raise SystemExit


summary = pd.DataFrame({
    "metric": [
        "Periods",
        "Mean correlation",
        "Median correlation",
        "P90 correlation",
        "Max correlation",
        "Pairs >= 0.50",
        "Pairs >= 0.70",
        "Pairs >= 0.80",
        "Correlation-adjusted Effective N",
    ],
    "value": [
        len(result),
        result["mean_corr"].mean(),
        result["median_corr"].mean(),
        result["p90_corr"].mean(),
        result["max_corr"].mean(),
        result["high_corr_50_pct"].mean(),
        result["high_corr_70_pct"].mean(),
        result["high_corr_80_pct"].mean(),
        result["effective_n_corr"].mean(),
    ]
})

print(
    summary.to_string(index=False)
)


# ------------------------------------------------------------
# 5. Highest correlation periods
# ------------------------------------------------------------

print()
print("=" * 70)
print("HIGHEST CORRELATION PERIODS")
print("=" * 70)

print(
    result
    .sort_values(
        "mean_corr",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 6. Lowest effective N periods
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOWEST CORRELATION-ADJUSTED EFFECTIVE N")
print("=" * 70)

print(
    result
    .sort_values(
        "effective_n_corr",
        ascending=True
    )
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 7. February 25, 2026
# ------------------------------------------------------------

target_date = pd.Timestamp("2026-02-25")

target = result[
    result["prediction_date"] == target_date
]

print()
print("=" * 70)
print("2026-02-25 CORRELATION ANALYSIS")
print("=" * 70)

if len(target):
    print(
        target.to_string(index=False)
    )
else:
    print(
        "2026-02-25 not found."
    )


# ------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------

output_path = (
    "data/correlation_risk_2026.csv"
)

result.to_csv(
    output_path,
    index=False
)

print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(output_path)

con.close()