import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/market.duckdb"

TARGET_DATE = pd.Timestamp("2026-02-25")
TOP_PCT = 0.20

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("2026-02-25 Alpha / Market Regime Failure Analysis")
print("=" * 70)


# ============================================================
# 1. Alpha predictions
# ============================================================

pred = con.sql(f"""
    SELECT
        date,
        symbol,
        prediction,
        future_excess_return_5d
    FROM alpha_lgbm_predictions
    WHERE date = '{TARGET_DATE.date()}'
      AND prediction IS NOT NULL
""").df()

pred["date"] = pd.to_datetime(pred["date"])

if len(pred) == 0:
    print("No prediction data.")
    con.close()
    raise SystemExit

pred = pred.sort_values(
    "prediction",
    ascending=False
).reset_index(drop=True)

n = len(pred)
n_top = max(1, int(np.floor(n * TOP_PCT)))

top = pred.head(n_top).copy()
bottom = pred.tail(n_top).copy()

print()
print(f"Universe           : {n}")
print(f"Top 20%            : {len(top)}")
print(f"Bottom 20%         : {len(bottom)}")


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
# 3. Trading dates
# ============================================================

trading_dates = sorted(
    pd.to_datetime(
        prices["date"].unique()
    )
)

entry_candidates = [
    d for d in trading_dates
    if d > TARGET_DATE
]

entry_date = entry_candidates[0]

entry_idx = trading_dates.index(
    entry_date
)

exit_date = trading_dates[
    entry_idx + 5
]

print()
print(f"Prediction date : {TARGET_DATE.date()}")
print(f"Entry date      : {entry_date.date()}")
print(f"Exit date       : {exit_date.date()}")


# ============================================================
# 4. Calculate 5-day open-to-open return
# ============================================================

entry = prices[
    prices["date"] == entry_date
][
    ["symbol", "open"]
].rename(
    columns={"open": "entry_open"}
)

exit_ = prices[
    prices["date"] == exit_date
][
    ["symbol", "open"]
].rename(
    columns={"open": "exit_open"}
)

realized = entry.merge(
    exit_,
    on="symbol",
    how="inner"
)

realized["realized_return"] = (
    realized["exit_open"]
    /
    realized["entry_open"]
    - 1
)


pred = pred.merge(
    realized[
        ["symbol", "realized_return"]
    ],
    on="symbol",
    how="left"
)

top = pred.head(n_top).copy()
bottom = pred.tail(n_top).copy()


# ============================================================
# 5. Cross-sectional Alpha analysis
# ============================================================

print()
print("=" * 70)
print("CROSS-SECTIONAL ALPHA ANALYSIS")
print("=" * 70)

def describe_group(name, df):

    print()
    print(name)

    print(
        f"Count                 : {len(df)}"
    )

    print(
        f"Mean prediction        : "
        f"{df['prediction'].mean():.6f}"
    )

    print(
        f"Median prediction      : "
        f"{df['prediction'].median():.6f}"
    )

    print(
        f"Mean realized return   : "
        f"{df['realized_return'].mean():.4%}"
    )

    print(
        f"Median realized return : "
        f"{df['realized_return'].median():.4%}"
    )

    print(
        f"Winning ratio          : "
        f"{(df['realized_return'] > 0).mean():.2%}"
    )

    print(
        f"Losing ratio           : "
        f"{(df['realized_return'] < 0).mean():.2%}"
    )

    print(
        f"Min realized return    : "
        f"{df['realized_return'].min():.4%}"
    )

    print(
        f"Max realized return    : "
        f"{df['realized_return'].max():.4%}"
    )


describe_group(
    "TOP 20%",
    top
)

describe_group(
    "BOTTOM 20%",
    bottom
)


# ============================================================
# 6. Top-minus-bottom spread
# ============================================================

top_mean = top[
    "realized_return"
].mean()

bottom_mean = bottom[
    "realized_return"
].mean()

spread = (
    top_mean
    - bottom_mean
)

print()
print("=" * 70)
print("TOP-BOTTOM SPREAD")
print("=" * 70)

print(
    f"Top 20% mean return    : "
    f"{top_mean:.4%}"
)

print(
    f"Bottom 20% mean return : "
    f"{bottom_mean:.4%}"
)

print(
    f"Top-Bottom spread      : "
    f"{spread:.4%}"
)


# ============================================================
# 7. Prediction vs realized correlation
# ============================================================

valid = pred.dropna(
    subset=[
        "prediction",
        "realized_return"
    ]
)

pearson = (
    valid["prediction"]
    .corr(
        valid["realized_return"],
        method="pearson"
    )
)

spearman = (
    valid["prediction"]
    .corr(
        valid["realized_return"],
        method="spearman"
    )
)

print()
print("=" * 70)
print("PREDICTION QUALITY")
print("=" * 70)

print(
    f"Pearson correlation  : "
    f"{pearson:.6f}"
)

print(
    f"Spearman correlation : "
    f"{spearman:.6f}"
)


# ============================================================
# 8. KOSPI regime
# ============================================================

kospi = con.sql("""
    SELECT
        date,
        close
    FROM kospi_daily
    ORDER BY date
""").df()

kospi["date"] = pd.to_datetime(
    kospi["date"]
)

kospi["ret_1d"] = (
    kospi["close"].pct_change()
)

kospi["ret_5d"] = (
    kospi["close"].pct_change(5)
)

kospi["ret_20d"] = (
    kospi["close"].pct_change(20)
)

kospi["vol_20d"] = (
    kospi["ret_1d"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

kospi_before = kospi[
    kospi["date"] <= TARGET_DATE
].tail(1)

print()
print("=" * 70)
print("KOSPI REGIME")
print("=" * 70)

if len(kospi_before):

    row = kospi_before.iloc[0]

    print(
        f"KOSPI close        : "
        f"{row['close']:.2f}"
    )

    print(
        f"KOSPI 1D return    : "
        f"{row['ret_1d']:.4%}"
    )

    print(
        f"KOSPI 5D return    : "
        f"{row['ret_5d']:.4%}"
    )

    print(
        f"KOSPI 20D return   : "
        f"{row['ret_20d']:.4%}"
    )

    print(
        f"KOSPI 20D vol      : "
        f"{row['vol_20d']:.4%}"
    )


# ============================================================
# 9. KOSPI actual return during holding period
# ============================================================

kospi_entry = kospi[
    kospi["date"] == entry_date
]

kospi_exit = kospi[
    kospi["date"] == exit_date
]

if len(kospi_entry) and len(kospi_exit):

    kospi_return = (
        kospi_exit.iloc[0]["close"]
        /
        kospi_entry.iloc[0]["close"]
        - 1
    )

    print(
        f"KOSPI holding return : "
        f"{kospi_return:.4%}"
    )


# ============================================================
# 10. Market-adjusted performance
# ============================================================

if len(kospi_entry) and len(kospi_exit):

    market_return = kospi_return

    top_market_adj = (
        top["realized_return"]
        - market_return
    ).mean()

    bottom_market_adj = (
        bottom["realized_return"]
        - market_return
    ).mean()

    print()
    print("=" * 70)
    print("MARKET-ADJUSTED PERFORMANCE")
    print("=" * 70)

    print(
        f"Top 20% excess vs KOSPI    : "
        f"{top_market_adj:.4%}"
    )

    print(
        f"Bottom 20% excess vs KOSPI : "
        f"{bottom_market_adj:.4%}"
    )


# ============================================================
# 11. Alpha ranking buckets
# ============================================================

print()
print("=" * 70)
print("PREDICTION DECILE ANALYSIS")
print("=" * 70)

pred["decile"] = pd.qcut(
    pred["prediction"],
    10,
    labels=False,
    duplicates="drop"
) + 1

decile = (
    pred
    .groupby("decile")
    .agg(
        count=("symbol", "count"),
        mean_prediction=("prediction", "mean"),
        mean_return=("realized_return", "mean"),
        median_return=("realized_return", "median"),
        win_rate=(
            "realized_return",
            lambda x: (x > 0).mean()
        ),
    )
    .reset_index()
)

print(
    decile.to_string(
        index=False
    )
)


# ============================================================
# 12. Save
# ============================================================

output_path = (
    "data/regime_failure_2026_02_25.csv"
)

pred.to_csv(
    output_path,
    index=False
)

print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(output_path)

con.close()
