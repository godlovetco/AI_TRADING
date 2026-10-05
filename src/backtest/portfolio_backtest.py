import duckdb
import numpy as np
import pandas as pd


DB_PATH = "data/market.duckdb"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PREDICTION_TABLE = "alpha_lgbm_predictions"

START_DATE = "2026-01-02"

TOP_PCT = 0.20
BOTTOM_PCT = 0.20

HOLDING_DAYS = 5

# One-way transaction cost
TRANSACTION_COST = 0.0010

INITIAL_CAPITAL = 100_000_000


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

print("=" * 70)
print("Portfolio Backtest")
print("=" * 70)

con = duckdb.connect(DB_PATH)

pred = con.execute(
    f"""
    SELECT
        symbol,
        date,
        prediction
    FROM {PREDICTION_TABLE}
    WHERE date >= '{START_DATE}'
    ORDER BY date, symbol
    """
).df()


# ------------------------------------------------------------
# Load price data
# ------------------------------------------------------------

prices = con.execute(
    f"""
    SELECT
        symbol,
        date,
        open,
        close
    FROM daily_prices
    WHERE date >= '{START_DATE}'
    ORDER BY symbol, date
    """
).df()

con.close()

pred["date"] = pd.to_datetime(pred["date"])
prices["date"] = pd.to_datetime(prices["date"])

print(f"Prediction rows : {len(pred):,}")
print(f"Price rows      : {len(prices):,}")
print(f"Symbols         : {pred['symbol'].nunique()}")


# ------------------------------------------------------------
# Build next-day execution prices
# ------------------------------------------------------------

prices = prices.sort_values(
    ["symbol", "date"]
).copy()

prices["entry_open"] = prices.groupby(
    "symbol"
)["open"].shift(-1)

prices["exit_open"] = prices.groupby(
    "symbol"
)["open"].shift(-(HOLDING_DAYS + 1))


prices_lookup = prices[
    [
        "symbol",
        "date",
        "entry_open",
        "exit_open",
    ]
].copy()


# ------------------------------------------------------------
# Merge prediction with execution prices
# ------------------------------------------------------------

trades = pred.merge(
    prices_lookup,
    on=["symbol", "date"],
    how="inner"
)

trades = trades.dropna(
    subset=["entry_open", "exit_open"]
).copy()


# ------------------------------------------------------------
# Determine cross-sectional ranks
# ------------------------------------------------------------

trades["rank_pct"] = trades.groupby(
    "date"
)["prediction"].rank(
    method="first",
    pct=True
)


trades["long"] = trades["rank_pct"] >= (
    1.0 - TOP_PCT
)

trades["short"] = trades["rank_pct"] <= (
    BOTTOM_PCT
)


# ------------------------------------------------------------
# Individual trade return
# ------------------------------------------------------------

trades["gross_return"] = (
    trades["exit_open"]
    / trades["entry_open"]
    - 1.0
)

# Round-trip transaction cost
trades["net_return"] = (
    trades["gross_return"]
    - 2.0 * TRANSACTION_COST
)


# ------------------------------------------------------------
# Daily portfolio returns
# ------------------------------------------------------------

long_daily = (
    trades[trades["long"]]
    .groupby("date")["net_return"]
    .mean()
    .rename("long_return")
)

short_daily = (
    trades[trades["short"]]
    .groupby("date")["net_return"]
    .mean()
    .rename("short_return")
)


daily = pd.concat(
    [
        long_daily,
        short_daily,
    ],
    axis=1
).fillna(0.0)


daily["long_short_return"] = (
    daily["long_return"]
    - daily["short_return"]
)


# ------------------------------------------------------------
# Performance function
# ------------------------------------------------------------

def performance_stats(returns):

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    cumulative = (
        1.0 + returns
    ).cumprod()

    total_return = cumulative.iloc[-1] - 1.0

    years = len(returns) / 252.0

    annual_return = (
        (1.0 + total_return) ** (1.0 / years) - 1.0
        if years > 0
        else np.nan
    )

    annual_vol = (
        returns.std() * np.sqrt(252)
    )

    sharpe = (
        annual_return / annual_vol
        if annual_vol > 0
        else np.nan
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max - 1.0
    )

    max_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    return {
        "trading_days": len(returns),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "daily_win_rate": win_rate,
    }


# ------------------------------------------------------------
# Print performance
# ------------------------------------------------------------

print()
print("=" * 70)
print("LONG-ONLY")
print("=" * 70)

long_stats = performance_stats(
    daily["long_return"]
)

for k, v in long_stats.items():

    if k == "trading_days":
        print(f"{k:22s}: {v}")

    else:
        print(f"{k:22s}: {v:.4f}")


print()
print("=" * 70)
print("LONG-SHORT")
print("=" * 70)

ls_stats = performance_stats(
    daily["long_short_return"]
)

for k, v in ls_stats.items():

    if k == "trading_days":
        print(f"{k:22s}: {v}")

    else:
        print(f"{k:22s}: {v:.4f}")


# ------------------------------------------------------------
# Trade statistics
# ------------------------------------------------------------

long_trades = trades[
    trades["long"]
]

short_trades = trades[
    trades["short"]
]


print()
print("=" * 70)
print("TRADE STATISTICS")
print("=" * 70)

print(
    f"Long trades        : {len(long_trades):,}"
)

print(
    f"Short trades       : {len(short_trades):,}"
)

print(
    f"Long mean gross    : "
    f"{long_trades['gross_return'].mean():.4f}"
)

print(
    f"Short mean gross   : "
    f"{short_trades['gross_return'].mean():.4f}"
)

print(
    f"Transaction cost    : "
    f"{TRANSACTION_COST:.4%} one-way"
)


# ------------------------------------------------------------
# Save backtest results
# ------------------------------------------------------------

con = duckdb.connect(DB_PATH)

con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_backtest_daily"
)

daily_reset = daily.reset_index()

con.register(
    "daily_reset",
    daily_reset
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_backtest_daily AS
    SELECT * FROM daily_reset
    """
)


con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_backtest_trades"
)

con.register(
    "trades",
    trades
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_backtest_trades AS
    SELECT * FROM trades
    """
)

con.close()


print()
print("=" * 70)
print("Saved")
print("=" * 70)

print("alpha_lgbm_backtest_daily")
print("alpha_lgbm_backtest_trades")