import duckdb
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"

PREDICTION_TABLE = "alpha_lgbm_predictions"
PRICE_TABLE = "daily_prices"

HOLDING_DAYS = 5

TOP_PCTS = [0.10, 0.20, 0.30]
TRANSACTION_COSTS = [0.0005, 0.0010, 0.0020]

TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


# ============================================================
# Helpers
# ============================================================

def max_drawdown(returns):
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def annualized_return(total_return, periods):
    """
    5 trading-day periods.
    Approximately 252 / 5 = 50.4 periods/year.
    """
    if periods <= 0:
        return np.nan

    annual_factor = 252.0 / HOLDING_DAYS

    if total_return <= -1.0:
        return -1.0

    return (1.0 + total_return) ** (annual_factor / periods) - 1.0


def annualized_volatility(period_returns):
    if len(period_returns) < 2:
        return np.nan

    return period_returns.std(ddof=1) * np.sqrt(252.0 / HOLDING_DAYS)


def sharpe_ratio(period_returns):
    if len(period_returns) < 2:
        return np.nan

    vol = period_returns.std(ddof=1)

    if vol == 0:
        return np.nan

    return (
        period_returns.mean()
        / vol
        * np.sqrt(252.0 / HOLDING_DAYS)
    )


# ============================================================
# Load data
# ============================================================

def load_data():

    con = duckdb.connect(str(DB_PATH), read_only=True)

    pred = con.execute(
        f"""
        SELECT
            symbol,
            date,
            prediction
        FROM {PREDICTION_TABLE}
        WHERE date >= '{TEST_START}'
          AND date <= '{TEST_END}'
        ORDER BY date, symbol
        """
    ).df()

    prices = con.execute(
        f"""
        SELECT
            symbol,
            date,
            open
        FROM {PRICE_TABLE}
        WHERE date >= '{TEST_START}'
          AND date <= '{TEST_END}'
        ORDER BY date, symbol
        """
    ).df()

    con.close()

    pred["date"] = pd.to_datetime(pred["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    return pred, prices


# ============================================================
# Single backtest
# ============================================================

def run_backtest(
    predictions,
    prices,
    top_pct,
    transaction_cost,
):

    price_pivot = prices.pivot(
        index="date",
        columns="symbol",
        values="open",
    ).sort_index()

    trading_dates = price_pivot.index.tolist()

    prediction_dates = sorted(
        predictions["date"].unique()
    )

    # Keep prediction dates for which enough future trading
    # dates exist for entry + 5-day holding.
    valid_dates = []

    for d in prediction_dates:

        try:
            idx = trading_dates.index(d)
        except ValueError:
            continue

        if idx + HOLDING_DAYS + 1 >= len(trading_dates):
            continue

        valid_dates.append(d)

    # Non-overlapping rebalance schedule.
    rebalance_dates = valid_dates[::HOLDING_DAYS]

    records = []

    for signal_date in rebalance_dates:

        try:
            signal_idx = trading_dates.index(signal_date)
        except ValueError:
            continue

        entry_idx = signal_idx + 1
        exit_idx = signal_idx + HOLDING_DAYS + 1

        if exit_idx >= len(trading_dates):
            continue

        entry_date = trading_dates[entry_idx]
        exit_date = trading_dates[exit_idx]

        day_pred = predictions[
            predictions["date"] == signal_date
        ].copy()

        day_pred = day_pred.dropna(
            subset=["prediction"]
        )

        if len(day_pred) < 10:
            continue

        # Remove symbols without both entry and exit prices.
        entry_prices = price_pivot.loc[
            entry_date
        ]

        exit_prices = price_pivot.loc[
            exit_date
        ]

        day_pred = day_pred[
            day_pred["symbol"].isin(entry_prices.index)
            & day_pred["symbol"].isin(exit_prices.index)
        ].copy()

        day_pred["entry_price"] = day_pred[
            "symbol"
        ].map(entry_prices)

        day_pred["exit_price"] = day_pred[
            "symbol"
        ].map(exit_prices)

        day_pred = day_pred.dropna(
            subset=["entry_price", "exit_price"]
        )

        if len(day_pred) < 10:
            continue

        n_stocks = len(day_pred)

        n_select = max(
            1,
            int(np.floor(n_stocks * top_pct))
        )

        day_pred = day_pred.sort_values(
            "prediction",
            ascending=False,
        )

        long_df = day_pred.head(n_select).copy()
        short_df = day_pred.tail(n_select).copy()

        # ----------------------------------------------------
        # Long return
        # ----------------------------------------------------

        long_df["gross_return"] = (
            long_df["exit_price"]
            / long_df["entry_price"]
            - 1.0
        )

        long_gross = long_df["gross_return"].mean()

        # One-way entry + one-way exit.
        long_net = (
            long_gross
            - 2.0 * transaction_cost
        )

        # ----------------------------------------------------
        # Short return
        # ----------------------------------------------------

        short_df["gross_return"] = (
            short_df["entry_price"]
            / short_df["exit_price"]
            - 1.0
        )

        short_gross = short_df["gross_return"].mean()

        short_net = (
            short_gross
            - 2.0 * transaction_cost
        )

        # Simple long-short spread.
        ls_gross = long_gross - short_gross

        # Long + short each incur entry/exit costs.
        ls_net = (
            ls_gross
            - 4.0 * transaction_cost
        )

        records.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "n_universe": n_stocks,
                "n_holdings": n_select,
                "long_gross": long_gross,
                "long_net": long_net,
                "short_gross": short_gross,
                "short_net": short_net,
                "ls_gross": ls_gross,
                "ls_net": ls_net,
            }
        )

    result = pd.DataFrame(records)

    return result


# ============================================================
# Statistics
# ============================================================

def summarize(result, top_pct, transaction_cost):

    if result.empty:
        return {
            "top_pct": top_pct,
            "cost": transaction_cost,
            "periods": 0,
        }

    long_returns = result["long_net"]
    ls_returns = result["ls_net"]

    long_total = (
        (1.0 + long_returns).prod() - 1.0
    )

    ls_total = (
        (1.0 + ls_returns).prod() - 1.0
    )

    return {
        "top_pct": top_pct,
        "cost": transaction_cost,
        "periods": len(result),

        "long_total_return": long_total,
        "long_annualized_return": annualized_return(
            long_total,
            len(result),
        ),
        "long_annualized_vol": annualized_volatility(
            long_returns
        ),
        "long_sharpe": sharpe_ratio(
            long_returns
        ),
        "long_mdd": max_drawdown(
            long_returns
        ),
        "long_win_rate": (
            long_returns > 0
        ).mean(),

        "long_mean_gross": result[
            "long_gross"
        ].mean(),

        "long_mean_net": result[
            "long_net"
        ].mean(),

        "ls_total_return": ls_total,
        "ls_annualized_return": annualized_return(
            ls_total,
            len(result),
        ),
        "ls_annualized_vol": annualized_volatility(
            ls_returns
        ),
        "ls_sharpe": sharpe_ratio(
            ls_returns
        ),
        "ls_mdd": max_drawdown(
            ls_returns
        ),
        "ls_win_rate": (
            ls_returns > 0
        ).mean(),

        "ls_mean_gross": result[
            "ls_gross"
        ].mean(),

        "ls_mean_net": result[
            "ls_net"
        ].mean(),

        "avg_holdings": result[
            "n_holdings"
        ].mean(),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("5-Day Non-Overlapping Sensitivity Analysis")
    print("=" * 80)

    predictions, prices = load_data()

    print(f"Prediction rows : {len(predictions):,}")
    print(f"Price rows      : {len(prices):,}")
    print(
        f"Prediction dates: "
        f"{predictions['date'].nunique():,}"
    )
    print(
        f"Symbols         : "
        f"{predictions['symbol'].nunique():,}"
    )

    all_results = []

    detailed_results = {}

    for top_pct in TOP_PCTS:

        for cost in TRANSACTION_COSTS:

            print(
                f"\nRunning "
                f"Top {top_pct:.0%}, "
                f"Cost {cost:.2%}..."
            )

            result = run_backtest(
                predictions,
                prices,
                top_pct,
                cost,
            )

            summary = summarize(
                result,
                top_pct,
                cost,
            )

            all_results.append(summary)

            detailed_results[
                (top_pct, cost)
            ] = result

    summary_df = pd.DataFrame(all_results)

    # --------------------------------------------------------
    # Pretty output
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("LONG-ONLY SENSITIVITY")
    print("=" * 80)

    display_cols = [
        "top_pct",
        "cost",
        "periods",
        "long_total_return",
        "long_annualized_return",
        "long_annualized_vol",
        "long_sharpe",
        "long_mdd",
        "long_win_rate",
        "long_mean_gross",
        "long_mean_net",
        "avg_holdings",
    ]

    print(
        summary_df[display_cols].to_string(
            index=False,
            formatters={
                "top_pct": lambda x: f"{x:.0%}",
                "cost": lambda x: f"{x:.2%}",
                "long_total_return": lambda x: f"{x:.2%}",
                "long_annualized_return": lambda x: f"{x:.2%}",
                "long_annualized_vol": lambda x: f"{x:.2%}",
                "long_sharpe": lambda x: f"{x:.3f}",
                "long_mdd": lambda x: f"{x:.2%}",
                "long_win_rate": lambda x: f"{x:.2%}",
                "long_mean_gross": lambda x: f"{x:.2%}",
                "long_mean_net": lambda x: f"{x:.2%}",
                "avg_holdings": lambda x: f"{x:.1f}",
            },
        )
    )

    print("\n")
    print("=" * 80)
    print("LONG-SHORT SENSITIVITY")
    print("=" * 80)

    ls_cols = [
        "top_pct",
        "cost",
        "periods",
        "ls_total_return",
        "ls_annualized_return",
        "ls_annualized_vol",
        "ls_sharpe",
        "ls_mdd",
        "ls_win_rate",
        "ls_mean_gross",
        "ls_mean_net",
    ]

    print(
        summary_df[ls_cols].to_string(
            index=False,
            formatters={
                "top_pct": lambda x: f"{x:.0%}",
                "cost": lambda x: f"{x:.2%}",
                "ls_total_return": lambda x: f"{x:.2%}",
                "ls_annualized_return": lambda x: f"{x:.2%}",
                "ls_annualized_vol": lambda x: f"{x:.2%}",
                "ls_sharpe": lambda x: f"{x:.3f}",
                "ls_mdd": lambda x: f"{x:.2%}",
                "ls_win_rate": lambda x: f"{x:.2%}",
                "ls_mean_gross": lambda x: f"{x:.2%}",
                "ls_mean_net": lambda x: f"{x:.2%}",
            },
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)

    summary_path = (
        output_dir
        / "alpha_lgbm_5day_sensitivity.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("Saved:")
    print(summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()