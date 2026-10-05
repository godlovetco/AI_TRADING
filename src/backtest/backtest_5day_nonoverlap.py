import duckdb
import numpy as np
import pandas as pd


DB_PATH = "data/market.duckdb"

PREDICTION_TABLE = "alpha_lgbm_predictions"

START_DATE = "2026-01-02"

TOP_PCT = 0.20
BOTTOM_PCT = 0.20

HOLDING_DAYS = 5

TRANSACTION_COST = 0.0010


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("Non-Overlapping 5-Day Portfolio Backtest")
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

prices = con.execute(
    """
    SELECT
        symbol,
        date,
        open,
        close
    FROM daily_prices
    ORDER BY symbol, date
    """
).df()

con.close()

pred["date"] = pd.to_datetime(pred["date"])
prices["date"] = pd.to_datetime(prices["date"])

print(f"Prediction rows : {len(pred):,}")
print(f"Price rows      : {len(prices):,}")
print(f"Symbols         : {pred['symbol'].nunique()}")


# ============================================================
# Trading calendar
# ============================================================

trading_dates = sorted(
    pred["date"].unique()
)

print(
    f"Prediction dates: "
    f"{len(trading_dates)}"
)


# ============================================================
# Price lookup
# ============================================================

prices = prices.sort_values(
    ["symbol", "date"]
).copy()

prices["next_open"] = prices.groupby(
    "symbol"
)["open"].shift(-1)

price_lookup = prices[
    [
        "symbol",
        "date",
        "open",
        "next_open",
    ]
].copy()


# ============================================================
# Build rebalance dates
# ============================================================

# We use every HOLDING_DAYS-th prediction date.
rebalance_dates = trading_dates[
    ::HOLDING_DAYS
]

print(
    f"Rebalance dates : "
    f"{len(rebalance_dates)}"
)


# ============================================================
# Portfolio simulation
# ============================================================

long_periods = []
ls_periods = []

trade_records = []


for date in rebalance_dates:

    current_date = pd.Timestamp(date)

    # --------------------------------------------------------
    # Find prediction
    # --------------------------------------------------------

    day_pred = pred[
        pred["date"] == current_date
    ].copy()

    if len(day_pred) < 20:
        continue

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    day_pred["rank_pct"] = day_pred[
        "prediction"
    ].rank(
        method="first",
        pct=True
    )

    long_symbols = day_pred[
        day_pred["rank_pct"] >= (1.0 - TOP_PCT)
    ]["symbol"].tolist()

    short_symbols = day_pred[
        day_pred["rank_pct"] <= BOTTOM_PCT
    ]["symbol"].tolist()

    # --------------------------------------------------------
    # Determine execution date
    # --------------------------------------------------------

    date_idx = trading_dates.index(date)

    entry_idx = date_idx + 1
    exit_idx = date_idx + HOLDING_DAYS + 1

    if exit_idx >= len(trading_dates):
        continue

    entry_date = pd.Timestamp(
        trading_dates[entry_idx]
    )

    exit_date = pd.Timestamp(
        trading_dates[exit_idx]
    )

    # --------------------------------------------------------
    # Load entry / exit prices
    # --------------------------------------------------------

    selected_symbols = (
        long_symbols + short_symbols
    )

    p = prices[
        prices["symbol"].isin(selected_symbols)
    ].copy()

    entry = p[
        p["date"] == entry_date
    ][
        ["symbol", "open"]
    ].rename(
        columns={"open": "entry_price"}
    )

    exit_ = p[
        p["date"] == exit_date
    ][
        ["symbol", "open"]
    ].rename(
        columns={"open": "exit_price"}
    )

    period = entry.merge(
        exit_,
        on="symbol",
        how="inner"
    )

    if len(period) < 10:
        continue

    period["gross_return"] = (
        period["exit_price"]
        / period["entry_price"]
        - 1.0
    )

    period["side"] = np.where(
        period["symbol"].isin(long_symbols),
        "long",
        "short"
    )

    # --------------------------------------------------------
    # Long portfolio
    # --------------------------------------------------------

    long_period = period[
        period["side"] == "long"
    ].copy()

    if len(long_period) > 0:

        long_gross = (
            long_period["gross_return"].mean()
        )

        # Buy + sell
        long_net = (
            long_gross
            - 2 * TRANSACTION_COST
        )

        long_periods.append(
            {
                "signal_date": current_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "gross_return": long_gross,
                "net_return": long_net,
                "n_stocks": len(long_period),
            }
        )

    # --------------------------------------------------------
    # Long-short portfolio
    # --------------------------------------------------------

    short_period = period[
        period["side"] == "short"
    ].copy()

    if (
        len(long_period) > 0
        and len(short_period) > 0
    ):

        long_gross = (
            long_period["gross_return"].mean()
        )

        short_gross = (
            short_period["gross_return"].mean()
        )

        # Short return = negative stock return
        long_short_gross = (
            long_gross
            - short_gross
        )

        # Long side:
        # buy + sell = 2 costs
        #
        # Short side:
        # short sell + cover = 2 costs
        #
        # Portfolio average:
        # approximately 2 transaction costs
        # on each side.
        long_short_net = (
            long_short_gross
            - 2 * TRANSACTION_COST
        )

        ls_periods.append(
            {
                "signal_date": current_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "long_gross": long_gross,
                "short_gross": short_gross,
                "gross_return": long_short_gross,
                "net_return": long_short_net,
                "n_long": len(long_period),
                "n_short": len(short_period),
            }
        )

    # --------------------------------------------------------
    # Trade records
    # --------------------------------------------------------

    for _, row in period.iterrows():

        trade_records.append(
            {
                "signal_date": current_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "symbol": row["symbol"],
                "side": row["side"],
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "gross_return": row["gross_return"],
            }
        )


# ============================================================
# Convert to DataFrames
# ============================================================

long_df = pd.DataFrame(
    long_periods
)

ls_df = pd.DataFrame(
    ls_periods
)

trades_df = pd.DataFrame(
    trade_records
)


# ============================================================
# Performance
# ============================================================

def calculate_performance(
    returns,
    periods_per_year=252 / HOLDING_DAYS
):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) == 0:
        return {}

    cumulative = (
        1.0 + returns
    ).cumprod()

    total_return = (
        cumulative.iloc[-1] - 1.0
    )

    annual_return = (
        (1.0 + total_return)
        ** (periods_per_year / len(returns))
        - 1.0
    )

    annual_volatility = (
        returns.std()
        * np.sqrt(periods_per_year)
    )

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
        - 1.0
    )

    max_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    return {
        "periods": len(returns),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


# ============================================================
# Print results
# ============================================================

print()
print("=" * 70)
print("LONG-ONLY")
print("=" * 70)

if not long_df.empty:

    long_stats = calculate_performance(
        long_df["net_return"]
    )

    for key, value in long_stats.items():

        if key == "periods":
            print(
                f"{key:22s}: {value}"
            )
        else:
            print(
                f"{key:22s}: {value:.4f}"
            )


print()
print("=" * 70)
print("LONG-SHORT")
print("=" * 70)

if not ls_df.empty:

    ls_stats = calculate_performance(
        ls_df["net_return"]
    )

    for key, value in ls_stats.items():

        if key == "periods":
            print(
                f"{key:22s}: {value}"
            )
        else:
            print(
                f"{key:22s}: {value:.4f}"
            )


# ============================================================
# Trade statistics
# ============================================================

print()
print("=" * 70)
print("TRADE STATISTICS")
print("=" * 70)

if not long_df.empty:

    print(
        f"Long periods       : "
        f"{len(long_df)}"
    )

    print(
        f"Mean long gross    : "
        f"{long_df['gross_return'].mean():.4f}"
    )

    print(
        f"Mean long net      : "
        f"{long_df['net_return'].mean():.4f}"
    )


if not ls_df.empty:

    print(
        f"Long-short periods : "
        f"{len(ls_df)}"
    )

    print(
        f"Mean LS gross      : "
        f"{ls_df['gross_return'].mean():.4f}"
    )

    print(
        f"Mean LS net        : "
        f"{ls_df['net_return'].mean():.4f}"
    )


print(
    f"Transaction cost    : "
    f"{TRANSACTION_COST:.4%} one-way"
)


# ============================================================
# Save
# ============================================================

con = duckdb.connect(DB_PATH)

con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_5day_long"
)

con.register(
    "long_df",
    long_df
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_5day_long AS
    SELECT * FROM long_df
    """
)


con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_5day_long_short"
)

con.register(
    "ls_df",
    ls_df
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_5day_long_short AS
    SELECT * FROM ls_df
    """
)


con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_5day_trades"
)

con.register(
    "trades_df",
    trades_df
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_5day_trades AS
    SELECT * FROM trades_df
    """
)

con.close()


print()
print("=" * 70)
print("Saved")
print("=" * 70)

print("alpha_lgbm_5day_long")
print("alpha_lgbm_5day_long_short")
print("alpha_lgbm_5day_trades")