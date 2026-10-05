import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/market.duckdb"

TARGET_DATE = pd.Timestamp("2026-02-25")
HOLD_DAYS = 5

TOP_PCT = 0.20
VOL_LOOKBACK = 20
MAX_POSITION = 0.04

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("2026-02-25 Loss Attribution Analysis")
print("=" * 70)


# ============================================================
# 1. Prediction
# ============================================================

pred = con.sql(f"""
    SELECT
        date AS prediction_date,
        symbol,
        prediction
    FROM alpha_lgbm_predictions
    WHERE date = '{TARGET_DATE.date()}'
      AND prediction IS NOT NULL
    ORDER BY prediction DESC
""").df()

if len(pred) == 0:
    print("No prediction data found.")
    con.close()
    raise SystemExit

print(f"Prediction rows : {len(pred):,}")

n_selected = max(
    1,
    int(np.floor(len(pred) * TOP_PCT))
)

selected = (
    pred
    .sort_values("prediction", ascending=False)
    .head(n_selected)
    .copy()
)

print(f"Top 20% selected: {len(selected)}")


# ============================================================
# 2. Price data
# ============================================================

prices = con.sql("""
    SELECT
        date,
        symbol,
        open,
        close
    FROM daily_prices
    ORDER BY symbol, date
""").df()

prices["date"] = pd.to_datetime(prices["date"])

prices["ret_1d"] = (
    prices
    .groupby("symbol")["close"]
    .pct_change()
)


# ============================================================
# 3. 20-day volatility
# ============================================================

prices["vol_20d"] = (
    prices
    .groupby("symbol")["ret_1d"]
    .transform(
        lambda x:
        x.rolling(VOL_LOOKBACK).std()
        * np.sqrt(252)
    )
)


vol_at_target = prices[
    prices["date"] < TARGET_DATE
].sort_values(
    ["symbol", "date"]
).groupby("symbol").tail(1)

vol_at_target = (
    vol_at_target[
        ["symbol", "vol_20d"]
    ]
    .dropna()
)

selected = selected.merge(
    vol_at_target,
    on="symbol",
    how="left"
)

selected = selected[
    selected["vol_20d"] > 0
].copy()


# ============================================================
# 4. Inverse-vol weighting
# ============================================================

def capped_inverse_vol_weights(
    vol_series,
    max_position=0.04
):
    inv_vol = 1.0 / vol_series

    weights = (
        inv_vol /
        inv_vol.sum()
    )

    weights = weights.clip(
        upper=max_position
    )

    # Redistribute excess weight
    for _ in range(100):

        excess = 1.0 - weights.sum()

        if excess <= 1e-12:
            break

        available = (
            max_position - weights
        )

        available = available[
            available > 1e-12
        ]

        if len(available) == 0:
            break

        inv_available = (
            1.0 /
            vol_series.loc[
                available.index
            ]
        )

        redistribution = (
            excess
            * inv_available
            / inv_available.sum()
        )

        weights.loc[
            redistribution.index
        ] += redistribution

        weights = weights.clip(
            upper=max_position
        )

    # Final normalization
    weights = weights / weights.sum()

    return weights


vol_series = (
    selected
    .set_index("symbol")["vol_20d"]
)

weights = capped_inverse_vol_weights(
    vol_series,
    MAX_POSITION
)

selected["weight"] = (
    selected["symbol"]
    .map(weights)
)


# ============================================================
# 5. Entry / Exit dates
# ============================================================

trading_dates = sorted(
    prices["date"].unique()
)

trading_dates = [
    pd.Timestamp(x)
    for x in trading_dates
]

entry_candidates = [
    d for d in trading_dates
    if d > TARGET_DATE
]

if not entry_candidates:
    print("No entry date found.")
    con.close()
    raise SystemExit

entry_date = entry_candidates[0]

entry_index = trading_dates.index(
    entry_date
)

exit_index = (
    entry_index + HOLD_DAYS
)

if exit_index >= len(trading_dates):
    print("No exit date found.")
    con.close()
    raise SystemExit

exit_date = trading_dates[exit_index]

print()
print(f"Prediction date : {TARGET_DATE.date()}")
print(f"Entry date      : {entry_date.date()}")
print(f"Exit date       : {exit_date.date()}")


# ============================================================
# 6. Entry / Exit prices
# ============================================================

entry_prices = prices[
    prices["date"] == entry_date
][
    ["symbol", "open"]
].rename(
    columns={"open": "entry_open"}
)

exit_prices = prices[
    prices["date"] == exit_date
][
    ["symbol", "open"]
].rename(
    columns={"open": "exit_open"}
)

selected = selected.merge(
    entry_prices,
    on="symbol",
    how="left"
)

selected = selected.merge(
    exit_prices,
    on="symbol",
    how="left"
)

selected = selected.dropna(
    subset=[
        "entry_open",
        "exit_open",
        "weight"
    ]
).copy()


# ============================================================
# 7. Individual returns / contribution
# ============================================================

selected["stock_return"] = (
    selected["exit_open"]
    /
    selected["entry_open"]
    - 1.0
)

selected["contribution"] = (
    selected["weight"]
    *
    selected["stock_return"]
)


# ============================================================
# 8. Portfolio result
# ============================================================

portfolio_gross = (
    selected["contribution"]
    .sum()
)

print()
print("=" * 70)
print("PORTFOLIO RESULT")
print("=" * 70)

print(
    f"Portfolio gross return: "
    f"{portfolio_gross:.4%}"
)

print(
    f"Number of positions   : "
    f"{len(selected)}"
)

print(
    f"Weight sum            : "
    f"{selected['weight'].sum():.6f}"
)

print(
    f"Max position          : "
    f"{selected['weight'].max():.4%}"
)


# ============================================================
# 9. Loss attribution
# ============================================================

selected["abs_contribution"] = (
    selected["contribution"].abs()
)

selected = selected.sort_values(
    "contribution"
)

print()
print("=" * 70)
print("WORST LOSS CONTRIBUTORS")
print("=" * 70)

display_cols = [
    "symbol",
    "prediction",
    "vol_20d",
    "weight",
    "entry_open",
    "exit_open",
    "stock_return",
    "contribution",
]

print(
    selected[
        display_cols
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 10. Best contributors
# ============================================================

print()
print("=" * 70)
print("BEST CONTRIBUTORS")
print("=" * 70)

print(
    selected[
        display_cols
    ]
    .sort_values(
        "contribution",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 11. Contribution concentration
# ============================================================

negative = selected[
    selected["contribution"] < 0
].copy()

negative_total = (
    negative["contribution"].sum()
)

if negative_total != 0:

    negative["loss_share"] = (
        negative["contribution"]
        /
        negative_total
    )

else:

    negative["loss_share"] = 0.0


print()
print("=" * 70)
print("LOSS CONCENTRATION")
print("=" * 70)

print(
    f"Negative contribution total : "
    f"{negative_total:.4%}"
)

print(
    f"Top 1 loss contribution      : "
    f"{negative['loss_share'].abs().sort_values(ascending=False).head(1).sum():.2%}"
)

print(
    f"Top 5 loss contribution      : "
    f"{negative['loss_share'].abs().sort_values(ascending=False).head(5).sum():.2%}"
)

print(
    f"Top 10 loss contribution     : "
    f"{negative['loss_share'].abs().sort_values(ascending=False).head(10).sum():.2%}"
)


# ============================================================
# 12. Prediction vs realized return
# ============================================================

print()
print("=" * 70)
print("ALPHA FAILURE CHECK")
print("=" * 70)

print(
    f"Mean prediction selected : "
    f"{selected['prediction'].mean():.6f}"
)

print(
    f"Mean realized return     : "
    f"{selected['stock_return'].mean():.4%}"
)

print(
    f"Median realized return   : "
    f"{selected['stock_return'].median():.4%}"
)

print(
    f"Winning stocks            : "
    f"{(selected['stock_return'] > 0).mean():.2%}"
)

print(
    f"Losing stocks             : "
    f"{(selected['stock_return'] < 0).mean():.2%}"
)


# ============================================================
# 13. Save
# ============================================================

output_path = (
    "data/loss_attribution_2026_02_25.csv"
)

selected.to_csv(
    output_path,
    index=False
)

print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(output_path)

con.close()
