import argparse
import duckdb
import numpy as np
import pandas as pd


# ============================================================
# Risk Model V1 Backtest
#
# Alpha:
#   alpha_lgbm_predictions
#
# Portfolio:
#   Top 20% Alpha
#   Inverse Volatility Scaling
#   Maximum Position = 4%
#   Maximum Rebalance Turnover = 30%
#
# Execution:
#   Prediction at T
#   Entry at next trading day's Open
#   Exit after 5 trading days at Open
#
# Transaction Cost:
#   0.10% x portfolio turnover
#
# Important:
#   Price history BEFORE test period is loaded so that
#   20-day volatility is available from the first test date.
# ============================================================


# ============================================================
# 1. Configuration
# ============================================================

DB_PATH = "data/market.duckdb"

TOP_PCT = 0.20
HOLDING_DAYS = 5

VOL_LOOKBACK = 20
BETA_LOOKBACK = 20
TARGET_BETA = 1.00
BETA_TOLERANCE = 0.01

MAX_POSITION = 0.04
MAX_TURNOVER = 0.30

TRANSACTION_COST = 0.001

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"


# ============================================================
# 2. Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Risk Model V1 Backtest"
)

parser.add_argument(
    "--start",
    default=DEFAULT_START,
    help="Backtest start date, e.g. 2025-01-01",
)

parser.add_argument(
    "--end",
    default=DEFAULT_END,
    help="Backtest end date, e.g. 2025-12-31",
)

args = parser.parse_args()


# ============================================================
# 3. Normalize dates
# ============================================================

TEST_START = pd.Timestamp(args.start)
TEST_END = pd.Timestamp(args.end)


# ============================================================
# 4. Helper functions
# ============================================================

def capped_inverse_vol_weights(
    vol_series: pd.Series,
    max_position: float,
) -> pd.Series:
    """
    Create inverse-volatility portfolio weights
    with a hard maximum position constraint.

    Properties:
        sum(weights) = 1
        max(weights) <= max_position
    """

    vol = vol_series.astype(float).copy()

    vol = vol.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    vol = vol.dropna()

    vol = vol[vol > 0]

    if vol.empty:
        raise ValueError(
            "No valid volatility values."
        )

    symbols = vol.index.tolist()

    inverse_vol = 1.0 / vol

    weights = pd.Series(
        0.0,
        index=symbols,
        dtype=float,
    )

    remaining_symbols = symbols.copy()
    remaining_capital = 1.0

    while remaining_symbols:

        active_inverse_vol = inverse_vol.loc[
            remaining_symbols
        ]

        proposed = (
            active_inverse_vol
            / active_inverse_vol.sum()
            * remaining_capital
        )

        over_cap = (
            proposed > max_position
        )

        if not over_cap.any():

            weights.loc[
                remaining_symbols
            ] = proposed

            break

        capped_symbols = proposed[
            over_cap
        ].index.tolist()

        for symbol in capped_symbols:

            weights.loc[symbol] = (
                max_position
            )

        remaining_capital -= (
            max_position
            * len(capped_symbols)
        )

        remaining_symbols = [
            symbol
            for symbol in remaining_symbols
            if symbol not in capped_symbols
        ]

        if remaining_capital <= 1e-12:

            break

    # --------------------------------------------------------
    # Numerical cleanup
    # --------------------------------------------------------

    weight_sum = weights.sum()

    if weight_sum <= 0:
        raise ValueError(
            "Portfolio weights sum to zero."
        )

    weights = weights / weight_sum

    # --------------------------------------------------------
    # Hard position-cap check
    # --------------------------------------------------------

    if weights.max() > max_position + 1e-8:

        raise RuntimeError(
            "Position cap violated: "
            f"{weights.max():.6f} "
            f"> {max_position:.6f}"
        )

    return weights



def adjust_beta_weights(
    weights: pd.Series,
    beta_series: pd.Series,
    target_beta: float,
    max_position: float,
):
    """
    Reduce portfolio beta toward target_beta while:

    1. keeping the portfolio fully invested,
    2. keeping weights non-negative,
    3. respecting max_position.

    The function transfers weight from higher-beta stocks
    to lower-beta stocks.

    Returns
    -------
    adjusted_weights : pd.Series
    portfolio_beta   : float
    beta_adjusted    : bool
    """

    weights = (
        weights
        .copy()
        .astype(float)
    )

    beta_series = (
        beta_series
        .reindex(weights.index)
        .astype(float)
    )

    valid = (
        weights.notna()
        & beta_series.notna()
        & np.isfinite(beta_series)
    )

    weights = weights[valid]
    beta_series = beta_series[valid]

    if weights.empty:
        return weights, np.nan, False

    weight_sum = weights.sum()

    if weight_sum <= 0:
        return weights, np.nan, False

    weights = weights / weight_sum

    def portfolio_beta(w):
        return float(
            (w * beta_series.reindex(w.index)).sum()
        )

    current_beta = portfolio_beta(weights)

    # Already within target
    if current_beta <= target_beta + BETA_TOLERANCE:
        return weights, current_beta, False

    adjusted = weights.copy()
    beta_adjusted = False

    # --------------------------------------------------------
    # Iteratively transfer weight:
    #
    # high-beta stock -> lowest-beta stock
    # --------------------------------------------------------

    for _ in range(100):

        current_beta = portfolio_beta(adjusted)

        if current_beta <= target_beta + BETA_TOLERANCE:
            break

        positive = adjusted[
            adjusted > 1e-12
        ].index

        if len(positive) == 0:
            break

        high_symbol = (
            beta_series.loc[positive]
            .idxmax()
        )

        high_beta = float(
            beta_series.loc[high_symbol]
        )

        # Candidate receiving stock:
        # lowest beta with remaining capacity.
        candidates = [
            s
            for s in adjusted.index
            if s != high_symbol
            and adjusted.loc[s] < max_position - 1e-10
            and beta_series.loc[s] < high_beta
        ]

        if not candidates:
            break

        low_symbol = min(
            candidates,
            key=lambda s: beta_series.loc[s],
        )

        low_beta = float(
            beta_series.loc[low_symbol]
        )

        beta_gap = high_beta - low_beta

        if beta_gap <= 1e-12:
            break

        # Weight transfer required to reach target beta.
        required_transfer = (
            current_beta - target_beta
        ) / beta_gap

        available_from_high = adjusted.loc[
            high_symbol
        ]

        available_to_low = (
            max_position
            - adjusted.loc[low_symbol]
        )

        transfer = min(
            required_transfer,
            available_from_high,
            available_to_low,
        )

        if transfer <= 1e-12:
            break

        adjusted.loc[high_symbol] -= transfer
        adjusted.loc[low_symbol] += transfer

        beta_adjusted = True

    # Numerical normalization
    adjusted = (
        adjusted
        / adjusted.sum()
    )

    final_beta = portfolio_beta(adjusted)

    return (
        adjusted,
        final_beta,
        beta_adjusted,
    )


def apply_turnover_constraint(
    target_weights: pd.Series,
    previous_weights: pd.Series,
    max_turnover: float,
):
    """
    Move previous portfolio toward target portfolio
    while keeping half-L1 turnover <= max_turnover.

    Turnover definition:

        turnover =
            0.5 * sum(abs(new_weight - old_weight))

    First portfolio entry is NOT handled here.
    """

    all_symbols = sorted(
        set(target_weights.index)
        | set(previous_weights.index)
    )

    target = target_weights.reindex(
        all_symbols,
        fill_value=0.0,
    )

    previous = previous_weights.reindex(
        all_symbols,
        fill_value=0.0,
    )

    raw_turnover = (
        np.abs(
            target - previous
        ).sum()
        / 2.0
    )

    # --------------------------------------------------------
    # No constraint needed
    # --------------------------------------------------------

    if raw_turnover <= max_turnover:

        return (
            target,
            raw_turnover,
            False,
        )

    # --------------------------------------------------------
    # Scale movement toward target
    # --------------------------------------------------------

    scale = (
        max_turnover
        / raw_turnover
    )

    constrained = (
        previous
        + scale * (
            target - previous
        )
    )

    # Numerical normalization
    constrained = (
        constrained
        / constrained.sum()
    )

    actual_turnover = (
        np.abs(
            constrained - previous
        ).sum()
        / 2.0
    )

    return (
        constrained,
        actual_turnover,
        True,
    )


def calculate_portfolio_return(
    weights: pd.Series,
    entry_prices: pd.Series,
    exit_prices: pd.Series,
) -> float:
    """
    Calculate weighted long-only portfolio return.
    """

    common_symbols = (
        weights.index
        .intersection(entry_prices.index)
        .intersection(exit_prices.index)
    )

    if len(common_symbols) == 0:

        return np.nan

    stock_returns = (
        exit_prices.loc[common_symbols]
        / entry_prices.loc[common_symbols]
        - 1.0
    )

    portfolio_return = (
        weights.loc[common_symbols]
        * stock_returns
    ).sum()

    return float(portfolio_return)


# ============================================================
# 5. Start
# ============================================================

print()
print("=" * 70)
print("Risk Model V2 Beta - Volatility + Beta + Position Cap + Turnover")
print("=" * 70)


# ============================================================
# 6. Connect DuckDB
# ============================================================

con = duckdb.connect(DB_PATH)


# ============================================================
# 7. Load Alpha Predictions
# ============================================================

pred = con.execute(
    """
    SELECT
        symbol,
        CAST(date AS DATE) AS date,
        prediction
    FROM alpha_lgbm_predictions
    WHERE date >= ?
      AND date <= ?
    ORDER BY date, symbol
    """,
    [
        TEST_START,
        TEST_END,
    ],
).fetchdf()


# ============================================================
# 8. Normalize prediction dates
# ============================================================

pred["date"] = pd.to_datetime(
    pred["date"]
)


# ============================================================
# 9. Load ALL price history through test end
#
# No lower bound.
#
# This is intentional.
#
# We need historical prices BEFORE TEST_START to calculate
# the 20-day volatility for the first test observations.
# ============================================================

prices = con.execute(
    """
    SELECT
        symbol,
        CAST(date AS DATE) AS date,
        open,
        high,
        low,
        close,
        volume
    FROM daily_prices
    WHERE date <= ?
    ORDER BY symbol, date
    """,
    [
        TEST_END,
    ],
).fetchdf()


# ============================================================
# 10. Normalize price dates
# ============================================================

prices["date"] = pd.to_datetime(
    prices["date"]
)


# ============================================================
# 10-B. Load KOSPI history
# ============================================================

kospi = con.execute(
    """
    SELECT
        CAST(date AS DATE) AS date,
        close
    FROM kospi_daily
    WHERE date <= ?
    ORDER BY date
    """,
    [
        TEST_END,
    ],
).fetchdf()

kospi["date"] = pd.to_datetime(
    kospi["date"]
)

kospi["kospi_ret_1d"] = (
    kospi["close"].pct_change()
)


# ============================================================
# 10-C. Merge KOSPI returns with stock prices
# ============================================================

prices = prices.merge(
    kospi[
        [
            "date",
            "kospi_ret_1d",
        ]
    ],
    on="date",
    how="left",
)


# ============================================================
# 11. Basic information
# ============================================================

print(
    f"Prediction rows : {len(pred):,}"
)

print(
    f"Price rows      : {len(prices):,}"
)

print(
    "Prediction dates: "
    f"{pred['date'].nunique():,}"
)

print(
    "Symbols         : "
    f"{pred['symbol'].nunique():,}"
)

print()
print("Risk Model Parameters")
print("-" * 70)

print(
    f"Top selection    : {TOP_PCT:.0%}"
)

print(
    f"Vol lookback     : {VOL_LOOKBACK} days"
)

print(
    f"Max position     : {MAX_POSITION:.2%}"
)

print(
    f"Beta lookback    : {BETA_LOOKBACK} days"
)

print(
    f"Target portfolio beta : {TARGET_BETA:.2f}"
)

print(
    f"Max turnover     : {MAX_TURNOVER:.2%}"
)

print(
    f"Transaction cost  : "
    f"{TRANSACTION_COST:.2%}"
)

print(
    f"Period           : "
    f"{TEST_START.date()} ~ {TEST_END.date()}"
)


# ============================================================
# 12. Calculate daily returns
# ============================================================

prices["ret_1d"] = (
    prices
    .groupby("symbol")["close"]
    .pct_change()
)


# ============================================================
# 12-B. Calculate 20-day rolling Beta vs KOSPI
#
# Beta =
#     Cov(stock_return, KOSPI_return)
#     --------------------------------
#     Var(KOSPI_return)
#
# ============================================================

prices["ret_kospi_product"] = (
    prices["ret_1d"]
    * prices["kospi_ret_1d"]
)

prices["stock_ret_mean_20d"] = (
    prices
    .groupby("symbol")["ret_1d"]
    .transform(
        lambda x:
            x.rolling(
                window=BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            ).mean()
    )
)

prices["kospi_ret_mean_20d"] = (
    prices
    .groupby("symbol")["kospi_ret_1d"]
    .transform(
        lambda x:
            x.rolling(
                window=BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            ).mean()
    )
)

prices["cross_mean_20d"] = (
    prices
    .groupby("symbol")["ret_kospi_product"]
    .transform(
        lambda x:
            x.rolling(
                window=BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            ).mean()
    )
)

prices["kospi_sq_mean_20d"] = (
    prices
    .groupby("symbol")["kospi_ret_1d"]
    .transform(
        lambda x:
            x.pow(2)
            .rolling(
                window=BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            ).mean()
    )
)

rolling_cov = (
    prices["cross_mean_20d"]
    - prices["stock_ret_mean_20d"]
    * prices["kospi_ret_mean_20d"]
)

rolling_var = (
    prices["kospi_sq_mean_20d"]
    - prices["kospi_ret_mean_20d"].pow(2)
)

prices["beta_20d"] = (
    rolling_cov
    / rolling_var
)

prices["beta_20d"] = (
    prices["beta_20d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


# ============================================================
# 13. Calculate 20-day rolling volatility
#
# This is calculated using historical data BEFORE test period
# as well.
#
# Annualization:
#     daily std * sqrt(252)
# ============================================================

prices["vol_20d"] = (
    prices
    .groupby("symbol")["ret_1d"]
    .transform(
        lambda x:
            x.rolling(
                window=VOL_LOOKBACK,
                min_periods=VOL_LOOKBACK,
            ).std()
    )
)

prices["vol_20d"] = (
    prices["vol_20d"]
    * np.sqrt(252.0)
)


# ============================================================
# 14. Keep volatility observations inside test period
# ============================================================

vol_data = prices[
    [
        "symbol",
        "date",
        "vol_20d",
    ]
].copy()


vol_data = vol_data[
    (vol_data["date"] >= TEST_START)
    & (vol_data["date"] <= TEST_END)
].copy()


# ============================================================
# 14-B. Keep Beta observations inside test period
# ============================================================

beta_data = prices[
    [
        "symbol",
        "date",
        "beta_20d",
    ]
].copy()

beta_data = beta_data[
    (beta_data["date"] >= TEST_START)
    & (beta_data["date"] <= TEST_END)
].copy()


# ============================================================
# 15. Merge alpha predictions with volatility and Beta
# ============================================================

pred = pred.merge(
    vol_data,
    on=[
        "symbol",
        "date",
    ],
    how="left",
)

pred = pred.merge(
    beta_data,
    on=[
        "symbol",
        "date",
    ],
    how="left",
)


# ============================================================
# 16. Remove invalid observations
# ============================================================

pred["vol_20d"] = (
    pred["vol_20d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

pred["beta_20d"] = (
    pred["beta_20d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


valid_pred = pred[
    pred["prediction"].notna()
    & pred["vol_20d"].notna()
    & (pred["vol_20d"] > 0)
    & pred["beta_20d"].notna()
].copy()


# ============================================================
# 17. Prediction dates
# ============================================================

prediction_dates = sorted(
    valid_pred["date"].unique()
)


if len(prediction_dates) == 0:

    con.close()

    raise RuntimeError(
        "No valid prediction dates."
    )


# ============================================================
# 18. Non-overlapping rebalance schedule
#
# Prediction date:
#     T
#
# Entry:
#     next available prediction/trading date
#
# Exit:
#     HOLDING_DAYS prediction dates later
# ============================================================

rebalance_dates = (
    prediction_dates[
        ::HOLDING_DAYS
    ]
)


# ============================================================
# 19. Price data for execution
# ============================================================

prices_eval = prices[
    (prices["date"] >= TEST_START)
    & (prices["date"] <= TEST_END)
].copy()


# ============================================================
# 20. Create date -> price table lookup
# ============================================================

price_dict = {
    date: group.set_index("symbol")
    for date, group
    in prices_eval.groupby("date")
}


# ============================================================
# 21. Portfolio simulation
# ============================================================

results = []

previous_weights = pd.Series(
    dtype=float
)

first_entry = True


# ============================================================
# 22. Backtest loop
# ============================================================

for prediction_date in rebalance_dates:

    prediction_date = pd.Timestamp(
        prediction_date
    )

    # --------------------------------------------------------
    # Find next trading date
    # --------------------------------------------------------

    future_dates = [
        date
        for date in prediction_dates
        if date > prediction_date
    ]

    if len(future_dates) == 0:
        continue

    entry_date = pd.Timestamp(
        future_dates[0]
    )

    # --------------------------------------------------------
    # Find exit date
    # --------------------------------------------------------

    entry_index = prediction_dates.index(
        entry_date
    )

    exit_index = (
        entry_index
        + HOLDING_DAYS
    )

    if exit_index >= len(prediction_dates):

        continue

    exit_date = pd.Timestamp(
        prediction_dates[
            exit_index
        ]
    )

    # --------------------------------------------------------
    # Alpha snapshot
    # --------------------------------------------------------

    day_pred = valid_pred[
        valid_pred["date"]
        == prediction_date
    ].copy()

    if day_pred.empty:

        continue

    # --------------------------------------------------------
    # Top 20% selection
    # --------------------------------------------------------

    n_selected = max(
        1,
        int(
            np.floor(
                len(day_pred)
                * TOP_PCT
            )
        ),
    )

    selected = (
        day_pred
        .sort_values(
            "prediction",
            ascending=False,
        )
        .head(n_selected)
        .copy()
    )

    selected = selected[
        selected["vol_20d"] > 0
    ].copy()

    if selected.empty:

        continue

    # --------------------------------------------------------
    # Inverse-volatility target weights
    # --------------------------------------------------------

    vol_series = (
        selected
        .set_index("symbol")[
            "vol_20d"
        ]
    )

    target_weights = (
        capped_inverse_vol_weights(
            vol_series,
            MAX_POSITION,
        )
    )

    # --------------------------------------------------------
    # Portfolio Beta adjustment
    #
    # Apply before the existing turnover constraint.
    # This preserves the V1 turnover mechanism.
    # --------------------------------------------------------

    beta_series = (
        selected
        .set_index("symbol")[
            "beta_20d"
        ]
    )

    (
        target_weights,
        target_portfolio_beta,
        beta_adjusted,
    ) = adjust_beta_weights(
        target_weights,
        beta_series,
        TARGET_BETA,
        MAX_POSITION,
    )

    # --------------------------------------------------------
    # First entry
    #
    # Initial portfolio is allowed to become fully invested.
    # No artificial 30% turnover restriction here.
    # --------------------------------------------------------

    if first_entry:

        final_weights = (
            target_weights.copy()
        )

        turnover = (
            final_weights.abs().sum()
        )

        turnover_constrained = False

        is_first_entry = True

        first_entry = False

    # --------------------------------------------------------
    # Subsequent rebalance
    # --------------------------------------------------------

    else:

        (
            final_weights,
            turnover,
            turnover_constrained,
        ) = apply_turnover_constraint(
            target_weights,
            previous_weights,
            MAX_TURNOVER,
        )

        is_first_entry = False

    # --------------------------------------------------------
    # Check execution dates
    # --------------------------------------------------------

    if entry_date not in price_dict:

        continue

    if exit_date not in price_dict:

        continue

    entry_prices = (
        price_dict[
            entry_date
        ]["open"]
    )

    exit_prices = (
        price_dict[
            exit_date
        ]["open"]
    )

    # --------------------------------------------------------
    # Only use stocks with both entry and exit prices
    # --------------------------------------------------------

    common_symbols = (
        final_weights.index
        .intersection(
            entry_prices.index
        )
        .intersection(
            exit_prices.index
        )
    )

    if len(common_symbols) == 0:

        continue

    final_weights = (
        final_weights
        .reindex(
            common_symbols
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Renormalize after price availability filtering
    # --------------------------------------------------------

    final_weights = (
        final_weights
        / final_weights.sum()
    )

    # --------------------------------------------------------
    # Position cap verification
    # --------------------------------------------------------

    if (
        final_weights.max()
        > MAX_POSITION + 1e-8
    ):

        con.close()

        raise RuntimeError(
            "Position cap violated "
            "after price filtering: "
            f"{final_weights.max():.6f}"
        )

    # --------------------------------------------------------
    # Gross portfolio return
    # --------------------------------------------------------

    gross_return = (
        calculate_portfolio_return(
            final_weights,
            entry_prices,
            exit_prices,
        )
    )

    if pd.isna(gross_return):

        continue

    # --------------------------------------------------------
    # Portfolio risk diagnostics
    # --------------------------------------------------------

    final_beta_series = (
        pred[
            (pred["date"] == prediction_date)
            & (
                pred["symbol"].isin(
                    final_weights.index
                )
            )
        ]
        .set_index("symbol")[
            "beta_20d"
        ]
    )

    final_beta_series = (
        final_beta_series
        .reindex(final_weights.index)
    )

    portfolio_beta = float(
        (
            final_weights
            * final_beta_series
        )
        .fillna(0.0)
        .sum()
    )

    hhi = float(
        (
            final_weights.pow(2)
        ).sum()
    )

    effective_n = (
        1.0 / hhi
        if hhi > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Transaction cost
    # --------------------------------------------------------

    transaction_cost = (
        turnover
        * TRANSACTION_COST
    )

    # --------------------------------------------------------
    # Net return
    # --------------------------------------------------------

    net_return = (
        gross_return
        - transaction_cost
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "prediction_date":
                prediction_date,

            "entry_date":
                entry_date,

            "exit_date":
                exit_date,

            "n_selected":
                len(common_symbols),

            "gross_return":
                gross_return,

            "turnover":
                turnover,

            "transaction_cost":
                transaction_cost,

            "net_return":
                net_return,

            "first_entry":
                is_first_entry,

            "turnover_constrained":
                turnover_constrained,

            "max_position":
                final_weights.max(),

            "portfolio_beta":
                portfolio_beta,

            "beta_adjusted":
                beta_adjusted,

            "HHI":
                hhi,

            "effective_n":
                effective_n,
        }
    )

    # --------------------------------------------------------
    # Save current portfolio
    # --------------------------------------------------------

    previous_weights = (
        final_weights.copy()
    )


# ============================================================
# 23. Convert results to DataFrame
# ============================================================

result_df = pd.DataFrame(
    results
)


if result_df.empty:

    con.close()

    raise RuntimeError(
        "No backtest results generated."
    )


# ============================================================
# 24. Performance calculations
# ============================================================

returns = (
    result_df["net_return"]
    .astype(float)
)

gross_returns = (
    result_df["gross_return"]
    .astype(float)
)


# ------------------------------------------------------------
# Total return
# ------------------------------------------------------------

total_return = (
    (1.0 + returns).prod()
    - 1.0
)


# ------------------------------------------------------------
# Number of periods
# ------------------------------------------------------------

periods = len(
    returns
)


# ------------------------------------------------------------
# Approximate elapsed years
#
# 5-trading-day holding period
# 252 trading days/year
# ------------------------------------------------------------

years = (
    periods
    * HOLDING_DAYS
    / 252.0
)


# ------------------------------------------------------------
# Annualized return
# ------------------------------------------------------------

if years > 0:

    annual_return = (
        (1.0 + total_return)
        ** (1.0 / years)
        - 1.0
    )

else:

    annual_return = np.nan


# ------------------------------------------------------------
# Annualized volatility
# ------------------------------------------------------------

if len(returns) > 1:

    annual_volatility = (
        returns.std(
            ddof=1
        )
        * np.sqrt(
            252.0
            / HOLDING_DAYS
        )
    )

else:

    annual_volatility = np.nan


# ------------------------------------------------------------
# Sharpe ratio
#
# Risk-free rate assumed 0 for this research stage.
# ------------------------------------------------------------

if (
    len(returns) > 1
    and returns.std(ddof=1) > 0
):

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(
            252.0
            / HOLDING_DAYS
        )
    )

else:

    sharpe = np.nan


# ------------------------------------------------------------
# Maximum drawdown
# ------------------------------------------------------------

equity = (
    1.0 + returns
).cumprod()

running_max = (
    equity.cummax()
)

drawdown = (
    equity
    / running_max
    - 1.0
)

max_drawdown = (
    drawdown.min()
)


# ------------------------------------------------------------
# Win rate
# ------------------------------------------------------------

win_rate = (
    returns > 0
).mean()


# ------------------------------------------------------------
# Mean returns
# ------------------------------------------------------------

mean_gross = (
    gross_returns.mean()
)

mean_net = (
    returns.mean()
)


# ------------------------------------------------------------
# Turnover
# ------------------------------------------------------------

mean_turnover_all = (
    result_df[
        "turnover"
    ].mean()
)


subsequent_results = (
    result_df[
        ~result_df[
            "first_entry"
        ]
    ]
)


if len(
    subsequent_results
) > 0:

    mean_rebalance_turnover = (
        subsequent_results[
            "turnover"
        ].mean()
    )

else:

    mean_rebalance_turnover = np.nan


# ------------------------------------------------------------
# Position cap
# ------------------------------------------------------------

max_position_observed = (
    result_df[
        "max_position"
    ].max()
)


# ============================================================
# 25. Print results
# ============================================================

print()
print("=" * 70)
print("RISK MODEL V2 BETA RESULTS")
print("=" * 70)

print(
    f"periods               : "
    f"{periods}"
)

print(
    f"total_return          : "
    f"{total_return:.4f}"
)

print(
    f"annual_return         : "
    f"{annual_return:.4f}"
)

print(
    f"annual_volatility     : "
    f"{annual_volatility:.4f}"
)

print(
    f"sharpe                : "
    f"{sharpe:.4f}"
)

print(
    f"max_drawdown          : "
    f"{max_drawdown:.4f}"
)

print(
    f"win_rate              : "
    f"{win_rate:.4f}"
)

print(
    f"mean_gross            : "
    f"{mean_gross:.4%}"
)

print(
    f"mean_net              : "
    f"{mean_net:.4%}"
)

print(
    f"mean_turnover_all     : "
    f"{mean_turnover_all:.4%}"
)

print(
    f"mean_rebalance_turnover:"
    f" {mean_rebalance_turnover:.4%}"
)

print(
    f"max_position_observed : "
    f"{max_position_observed:.4%}"
)


# ============================================================
# 26. First 10 periods
# ============================================================

print()
print("=" * 70)
print("First 10 periods")
print("=" * 70)

display_columns = [
    "prediction_date",
    "entry_date",
    "exit_date",
    "n_selected",
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "turnover_constrained",
    "max_position",
]

print(
    result_df[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 27. Save results to DuckDB
# ============================================================

table_name = (
    "alpha_lgbm_risk_v1_backtest"
)

con.execute(
    f"""
    CREATE OR REPLACE TABLE
    {table_name}
    AS
    SELECT *
    FROM result_df
    """
)


# ============================================================
# 28. Close database
# ============================================================

con.close()


# ============================================================
# 29. Finished
# ============================================================

print()
print("=" * 70)
print("Saved")
print("=" * 70)

print(
    table_name
)

print()