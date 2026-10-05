import argparse
import duckdb
import numpy as np
import pandas as pd


# ============================================================
# Risk V2 + Regime Filter
#
# Alpha:
#   LightGBM
#
# Portfolio:
#   Top 20%
#   20D Inverse Vol
#   4% Position Cap
#   60D Beta
#   30% Turnover Constraint
#
# Regime Overlay:
#   BASELINE             -> 100%
#   HIGH_VOL_RALLY_50    -> 50%
#   COMPOSITE_STRESS_25  -> 25%
#
# Cash is explicitly represented as "__CASH__".
# ============================================================

DB_PATH = "data/market.duckdb"

TOP_PCT = 0.20
HOLDING_DAYS = 5

VOL_LOOKBACK = 20
BETA_LOOKBACK = 60

TARGET_BETA = 1.00
BETA_TOLERANCE = 0.01

MAX_POSITION = 0.04
MAX_TURNOVER = 0.30

TRANSACTION_COST = 0.001

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-09-30"

REGIME_FILE = "data/alpha_regime_analysis.csv"

OUTPUT_TABLE = "alpha_lgbm_risk_v3_regime_backtest"


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--start",
    default=DEFAULT_START,
)

parser.add_argument(
    "--end",
    default=DEFAULT_END,
)

args = parser.parse_args()

TEST_START = pd.Timestamp(args.start)
TEST_END = pd.Timestamp(args.end)


# ============================================================
# Inverse Volatility
# ============================================================

def capped_inverse_vol_weights(
    vol_series,
    max_position,
):

    vol = (
        vol_series
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    vol = vol[vol > 0]

    if vol.empty:
        raise ValueError(
            "No valid volatility values."
        )

    inverse_vol = 1.0 / vol

    weights = pd.Series(
        0.0,
        index=vol.index,
        dtype=float,
    )

    remaining_symbols = list(vol.index)

    remaining_capital = 1.0

    while remaining_symbols:

        active_inverse_vol = (
            inverse_vol
            .loc[remaining_symbols]
        )

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

        capped_symbols = (
            proposed[over_cap]
            .index
            .tolist()
        )

        for symbol in capped_symbols:

            weights.loc[symbol] = (
                max_position
            )

        remaining_capital -= (
            max_position
            * len(capped_symbols)
        )

        remaining_symbols = [
            s
            for s in remaining_symbols
            if s not in capped_symbols
        ]

        if remaining_capital <= 1e-12:
            break

    if weights.sum() <= 0:
        raise ValueError(
            "Portfolio weights sum to zero."
        )

    weights = (
        weights
        / weights.sum()
    )

    if (
        weights.max()
        > max_position + 1e-8
    ):

        raise RuntimeError(
            "Position cap violated."
        )

    return weights


# ============================================================
# Beta Adjustment
# ============================================================

def adjust_beta_weights(
    weights,
    beta_series,
    target_beta,
    max_position,
):

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

    weights = (
        weights
        / weights.sum()
    )

    def portfolio_beta(w):

        return float(
            (
                w
                * beta_series.reindex(
                    w.index
                )
            ).sum()
        )

    current_beta = (
        portfolio_beta(weights)
    )

    if (
        current_beta
        <= target_beta
        + BETA_TOLERANCE
    ):

        return (
            weights,
            current_beta,
            False,
        )

    adjusted = weights.copy()

    beta_adjusted = False

    for _ in range(100):

        current_beta = (
            portfolio_beta(adjusted)
        )

        if (
            current_beta
            <= target_beta
            + BETA_TOLERANCE
        ):
            break

        positive = (
            adjusted[
                adjusted > 1e-12
            ].index
        )

        if len(positive) == 0:
            break

        high_symbol = (
            beta_series
            .loc[positive]
            .idxmax()
        )

        high_beta = float(
            beta_series.loc[
                high_symbol
            ]
        )

        candidates = [
            s
            for s in adjusted.index
            if (
                s != high_symbol
                and adjusted.loc[s]
                < max_position - 1e-10
                and beta_series.loc[s]
                < high_beta
            )
        ]

        if not candidates:
            break

        low_symbol = min(
            candidates,
            key=lambda s:
                beta_series.loc[s],
        )

        low_beta = float(
            beta_series.loc[
                low_symbol
            ]
        )

        beta_gap = (
            high_beta
            - low_beta
        )

        if beta_gap <= 1e-12:
            break

        required_transfer = (
            current_beta
            - target_beta
        ) / beta_gap

        available_from_high = (
            adjusted.loc[
                high_symbol
            ]
        )

        available_to_low = (
            max_position
            - adjusted.loc[
                low_symbol
            ]
        )

        transfer = min(
            required_transfer,
            available_from_high,
            available_to_low,
        )

        if transfer <= 1e-12:
            break

        adjusted.loc[
            high_symbol
        ] -= transfer

        adjusted.loc[
            low_symbol
        ] += transfer

        beta_adjusted = True

    adjusted = (
        adjusted
        / adjusted.sum()
    )

    final_beta = (
        portfolio_beta(adjusted)
    )

    return (
        adjusted,
        final_beta,
        beta_adjusted,
    )


# ============================================================
# Turnover
# ============================================================

def apply_turnover_constraint(
    target_weights,
    previous_weights,
    max_turnover,
):

    all_symbols = sorted(
        set(target_weights.index)
        | set(previous_weights.index)
    )

    target = (
        target_weights
        .reindex(
            all_symbols,
            fill_value=0.0,
        )
    )

    previous = (
        previous_weights
        .reindex(
            all_symbols,
            fill_value=0.0,
        )
    )

    raw_turnover = (
        np.abs(
            target - previous
        ).sum()
        / 2.0
    )

    if raw_turnover <= max_turnover:

        return (
            target,
            raw_turnover,
            False,
        )

    scale = (
        max_turnover
        / raw_turnover
    )

    constrained = (
        previous
        + scale
        * (
            target
            - previous
        )
    )

    constrained = (
        constrained
        / constrained.sum()
    )

    actual_turnover = (
        np.abs(
            constrained
            - previous
        ).sum()
        / 2.0
    )

    return (
        constrained,
        actual_turnover,
        True,
    )


# ============================================================
# Portfolio Return
# ============================================================

def calculate_portfolio_return(
    weights,
    entry_prices,
    exit_prices,
):

    stock_weights = (
        weights
        .drop(
            labels=["__CASH__"],
            errors="ignore",
        )
    )

    common = (
        stock_weights.index
        .intersection(
            entry_prices.index
        )
        .intersection(
            exit_prices.index
        )
    )

    if len(common) == 0:
        return np.nan

    stock_returns = (
        exit_prices.loc[common]
        / entry_prices.loc[common]
        - 1.0
    )

    return float(
        (
            stock_weights.loc[common]
            * stock_returns
        ).sum()
    )


# ============================================================
# Regime Exposure
# ============================================================

def get_exposure(
    row,
    strategy,
):

    if strategy == "V2_BASELINE":
        return 1.0

    stress = str(
        row.get(
            "stress_regime",
            "NORMAL",
        )
    )

    ret_5d = float(
        row.get(
            "ret_5d",
            np.nan,
        )
    )

    ret_20d = float(
        row.get(
            "ret_20d",
            np.nan,
        )
    )

    vol_20d = float(
        row.get(
            "vol_20d",
            np.nan,
        )
    )

    breadth = float(
        row.get(
            "breadth_ma60",
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Level 1:
    # HIGH VOL RALLY
    #
    # 고변동성 상승장에서는 exposure 50%
    # --------------------------------------------------------

    high_vol_rally = (
        stress == "HIGH_VOL_RALLY"
    )

    # --------------------------------------------------------
    # Level 2:
    # Severe Stress
    #
    # HIGH_VOL_RALLY와 별도로 정의
    # --------------------------------------------------------

    severe_breadth_stress = (
        np.isfinite(vol_20d)
        and np.isfinite(breadth)
        and vol_20d >= 0.40
        and breadth <= 0.30
    )

    severe_downtrend = (
        np.isfinite(ret_20d)
        and ret_20d <= -0.15
    )

    reversal = (
        np.isfinite(ret_20d)
        and np.isfinite(ret_5d)
        and ret_20d >= 0.10
        and ret_5d <= -0.05
    )

    severe_stress = (
        severe_breadth_stress
        or severe_downtrend
        or reversal
    )

    # --------------------------------------------------------
    # Strategy 1
    # --------------------------------------------------------

    if strategy == "HIGH_VOL_RALLY_50":

        return (
            0.50
            if high_vol_rally
            else 1.00
        )

    # --------------------------------------------------------
    # Strategy 2
    #
    # 100% normal
    # 50% high-vol rally
    # 25% severe stress
    # --------------------------------------------------------

    if strategy == "TWO_STAGE_50_25":

        if severe_stress:
            return 0.25

        if high_vol_rally:
            return 0.50

        return 1.00

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# Start
# ============================================================

print()
print("=" * 72)
print(
    "Risk V3 Two-Stage Regime Backtest"
)
print("=" * 72)


con = duckdb.connect(
    DB_PATH
)


# ============================================================
# Alpha
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

pred["date"] = pd.to_datetime(
    pred["date"]
)


# ============================================================
# Prices
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

prices["date"] = pd.to_datetime(
    prices["date"]
)


# ============================================================
# KOSPI
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


print(
    f"Prediction rows : "
    f"{len(pred):,}"
)

print(
    f"Price rows      : "
    f"{len(prices):,}"
)

print(
    f"Prediction dates: "
    f"{pred['date'].nunique():,}"
)

print(
    f"Symbols         : "
    f"{pred['symbol'].nunique():,}"
)


# ============================================================
# Returns
# ============================================================

prices["ret_1d"] = (
    prices
    .groupby("symbol")[
        "close"
    ]
    .pct_change()
)


# ============================================================
# 60D Beta
# ============================================================

prices["ret_kospi_product"] = (
    prices["ret_1d"]
    * prices["kospi_ret_1d"]
)

prices[
    "stock_ret_mean_beta"
] = (
    prices
    .groupby("symbol")[
        "ret_1d"
    ]
    .transform(
        lambda x:
        x.rolling(
            BETA_LOOKBACK,
            min_periods=
            BETA_LOOKBACK,
        ).mean()
    )
)

prices[
    "kospi_ret_mean_beta"
] = (
    prices
    .groupby("symbol")[
        "kospi_ret_1d"
    ]
    .transform(
        lambda x:
        x.rolling(
            BETA_LOOKBACK,
            min_periods=
            BETA_LOOKBACK,
        ).mean()
    )
)

prices[
    "cross_mean_beta"
] = (
    prices
    .groupby("symbol")[
        "ret_kospi_product"
    ]
    .transform(
        lambda x:
        x.rolling(
            BETA_LOOKBACK,
            min_periods=
            BETA_LOOKBACK,
        ).mean()
    )
)

prices[
    "kospi_sq_mean_beta"
] = (
    prices
    .groupby("symbol")[
        "kospi_ret_1d"
    ]
    .transform(
        lambda x:
        x.pow(2)
        .rolling(
            BETA_LOOKBACK,
            min_periods=
            BETA_LOOKBACK,
        ).mean()
    )
)

rolling_cov = (
    prices[
        "cross_mean_beta"
    ]
    - prices[
        "stock_ret_mean_beta"
    ]
    * prices[
        "kospi_ret_mean_beta"
    ]
)

rolling_var = (
    prices[
        "kospi_sq_mean_beta"
    ]
    - prices[
        "kospi_ret_mean_beta"
    ].pow(2)
)

prices["beta_60d"] = (
    rolling_cov
    / rolling_var
)

prices["beta_60d"] = (
    prices["beta_60d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


# ============================================================
# 20D Volatility
# ============================================================

prices["vol_20d"] = (
    prices
    .groupby("symbol")[
        "ret_1d"
    ]
    .transform(
        lambda x:
        x.rolling(
            VOL_LOOKBACK,
            min_periods=
            VOL_LOOKBACK,
        ).std()
    )
    * np.sqrt(252.0)
)


# ============================================================
# Merge Risk Data
# ============================================================

risk_data = prices[
    [
        "symbol",
        "date",
        "vol_20d",
        "beta_60d",
    ]
].copy()

risk_data = risk_data[
    (risk_data["date"] >= TEST_START)
    & (risk_data["date"] <= TEST_END)
]

pred = pred.merge(
    risk_data,
    on=[
        "symbol",
        "date",
    ],
    how="left",
)

pred["vol_20d"] = (
    pred["vol_20d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

pred["beta_60d"] = (
    pred["beta_60d"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

valid_pred = pred[
    pred["prediction"].notna()
    & pred["vol_20d"].notna()
    & (pred["vol_20d"] > 0)
    & pred["beta_60d"].notna()
].copy()


prediction_dates = sorted(
    valid_pred["date"].unique()
)

# ============================================================
# Canonical Unified Rebalance Schedule
# ============================================================

SCHEDULE_FILE = (
    "data/risk_unified_rebalance_schedule.csv"
)

schedule_df = pd.read_csv(
    SCHEDULE_FILE
)

schedule_df["date"] = pd.to_datetime(
    schedule_df["date"]
)

rebalance_dates = [
    pd.Timestamp(d)
    for d in schedule_df["date"]
    if (
        pd.Timestamp(d) >= TEST_START
        and pd.Timestamp(d) <= TEST_END
    )
]


# ============================================================
# Price lookup
# ============================================================

prices_eval = prices[
    (prices["date"] >= TEST_START)
    & (prices["date"] <= TEST_END)
].copy()

price_dict = {
    date: group.set_index(
        "symbol"
    )
    for date, group
    in prices_eval.groupby(
        "date"
    )
}


# ============================================================
# Regime Data
# ============================================================

regime = pd.read_csv(
    REGIME_FILE
)

if "date" in regime.columns:

    regime["date"] = pd.to_datetime(
        regime["date"]
    )

elif "prediction_date" in regime.columns:

    regime["date"] = pd.to_datetime(
        regime["prediction_date"]
    )

else:

    con.close()

    raise RuntimeError(
        "No date column found in "
        f"{REGIME_FILE}"
    )

regime = regime[
    (regime["date"] >= TEST_START)
    & (regime["date"] <= TEST_END)
].copy()

regime_dict = (
    regime
    .set_index("date")
    .to_dict("index")
)


# ============================================================
# Strategies
# ============================================================

strategies = [
    "V2_BASELINE",
    "HIGH_VOL_RALLY_50",
    "TWO_STAGE_50_25",
]


all_results = []


# ============================================================
# Backtest
# ============================================================

for strategy in strategies:

    print()
    print(
        f"Running: {strategy}"
    )

    previous_weights = (
        pd.Series(dtype=float)
    )

    first_entry = True

    for prediction_date in (
        rebalance_dates
    ):

        prediction_date = (
            pd.Timestamp(
                prediction_date
            )
        )

        future_dates = [
            d
            for d in prediction_dates
            if d > prediction_date
        ]

        if not future_dates:
            continue

        entry_date = pd.Timestamp(
            future_dates[0]
        )

        entry_index = (
            prediction_dates.index(
                entry_date
            )
        )

        exit_index = (
            entry_index
            + HOLDING_DAYS
        )

        if (
            exit_index
            >= len(prediction_dates)
        ):
            continue

        exit_date = pd.Timestamp(
            prediction_dates[
                exit_index
            ]
        )

        day_pred = valid_pred[
            valid_pred["date"]
            == prediction_date
        ].copy()

        if day_pred.empty:
            continue

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

        if selected.empty:
            continue


        # ----------------------------------------------------
        # Inverse Vol
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Beta
        # ----------------------------------------------------

        beta_series = (
            selected
            .set_index("symbol")[
                "beta_60d"
            ]
        )

        (
            target_weights,
            target_beta,
            beta_adjusted,
        ) = adjust_beta_weights(
            target_weights,
            beta_series,
            TARGET_BETA,
            MAX_POSITION,
        )


        # ----------------------------------------------------
        # Regime
        # ----------------------------------------------------

        regime_row = (
            regime_dict.get(
                prediction_date,
                {},
            )
        )

        exposure = get_exposure(
            regime_row,
            strategy,
        )


        # ----------------------------------------------------
        # Explicit cash
        # ----------------------------------------------------

        target_weights = (
            target_weights
            * exposure
        )

        target_weights[
            "__CASH__"
        ] = (
            1.0
            - exposure
        )


        # ----------------------------------------------------
        # First Entry
        #
        # Match original V2:
        # first entry turnover = 1.0
        # ----------------------------------------------------

        if first_entry:

            final_weights = (
                target_weights.copy()
            )

            turnover = 1.0

            turnover_constrained = (
                False
            )

            is_first_entry = True

            first_entry = False


        # ----------------------------------------------------
        # Rebalance
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------

        if (
            entry_date
            not in price_dict
        ):
            continue

        if (
            exit_date
            not in price_dict
        ):
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


        # ----------------------------------------------------
        # Price availability
        # ----------------------------------------------------

        stock_symbols = (
            final_weights.index[
                final_weights.index
                != "__CASH__"
            ]
        )

        common_symbols = (
            stock_symbols
            .intersection(
                entry_prices.index
            )
            .intersection(
                exit_prices.index
            )
        )

        if len(common_symbols) == 0:
            continue


        # ----------------------------------------------------
        # Missing stocks -> cash
        # ----------------------------------------------------

        available_weights = (
            final_weights
            .reindex(
                common_symbols,
                fill_value=0.0,
            )
        )

        original_stock_weight = (
            final_weights
            .reindex(
                stock_symbols,
                fill_value=0.0,
            )
            .sum()
        )

        available_stock_weight = (
            available_weights.sum()
        )

        missing_stock_weight = (
            original_stock_weight
            - available_stock_weight
        )

        cash_weight = (
            float(
                final_weights.get(
                    "__CASH__",
                    0.0,
                )
            )
            + max(
                0.0,
                missing_stock_weight,
            )
        )

        final_weights = (
            available_weights.copy()
        )

        final_weights[
            "__CASH__"
        ] = cash_weight

        final_weights = (
            final_weights
            / final_weights.sum()
        )


        # ----------------------------------------------------
        # Position Cap
        # ----------------------------------------------------

        stock_weights = (
            final_weights
            .drop(
                labels=["__CASH__"],
                errors="ignore",
            )
        )

        max_position = (
            stock_weights.max()
            if not stock_weights.empty
            else 0.0
        )

        if (
            max_position
            > MAX_POSITION + 1e-8
        ):

            con.close()

            raise RuntimeError(
                "Position cap violated: "
                f"{max_position:.6f}"
            )


        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        gross_return = (
            calculate_portfolio_return(
                final_weights,
                entry_prices,
                exit_prices,
            )
        )

        if pd.isna(gross_return):
            continue


        # ----------------------------------------------------
        # Final Beta
        # ----------------------------------------------------

        final_beta_series = (
            pred[
                (pred["date"]
                 == prediction_date)
                & pred["symbol"].isin(
                    stock_weights.index
                )
            ]
            .set_index("symbol")[
                "beta_60d"
            ]
            .reindex(
                stock_weights.index
            )
        )

        portfolio_beta = float(
            (
                stock_weights
                * final_beta_series
            )
            .fillna(0.0)
            .sum()
        )


        # ----------------------------------------------------
        # Concentration
        # ----------------------------------------------------

        hhi = float(
            stock_weights.pow(2).sum()
        )

        effective_n = (
            1.0 / hhi
            if hhi > 0
            else np.nan
        )


        # ----------------------------------------------------
        # Cost
        # ----------------------------------------------------

        transaction_cost = (
            turnover
            * TRANSACTION_COST
        )

        net_return = (
            gross_return
            - transaction_cost
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        all_results.append(
            {
                "strategy":
                    strategy,

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

                "exposure":
                    exposure,

                "cash_weight":
                    float(
                        final_weights.get(
                            "__CASH__",
                            0.0,
                        )
                    ),

                "turnover_constrained":
                    turnover_constrained,

                "max_position":
                    max_position,

                "portfolio_beta":
                    portfolio_beta,

                "target_beta":
                    target_beta,

                "beta_adjusted":
                    beta_adjusted,

                "HHI":
                    hhi,

                "effective_n":
                    effective_n,

                "stress_regime":
                    regime_row.get(
                        "stress_regime",
                        "UNKNOWN",
                    ),

                "vol_regime":
                    regime_row.get(
                        "vol_regime",
                        "UNKNOWN",
                    ),

                "ret_5d":
                    regime_row.get(
                        "ret_5d",
                        np.nan,
                    ),

                "ret_20d":
                    regime_row.get(
                        "ret_20d",
                        np.nan,
                    ),

                "vol_20d":
                    regime_row.get(
                        "vol_20d",
                        np.nan,
                    ),

                "breadth_ma60":
                    regime_row.get(
                        "breadth_ma60",
                        np.nan,
                    ),
            }
        )


        previous_weights = (
            final_weights.copy()
        )


con.close()


# ============================================================
# Results
# ============================================================

result_df = pd.DataFrame(
    all_results
)

if result_df.empty:

    raise RuntimeError(
        "No backtest results generated."
    )


# ============================================================
# Performance Function
# ============================================================

def performance(df):

    if df.empty:

        return {
            "periods": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "mean_gross": np.nan,
            "mean_net": np.nan,
            "mean_turnover": np.nan,
            "mean_exposure": np.nan,
            "reduced_ratio": np.nan,
            "max_position": np.nan,
            "mean_beta": np.nan,
        }

    returns = (
        df["net_return"]
        .astype(float)
    )

    gross_returns = (
        df["gross_return"]
        .astype(float)
    )

    periods = len(
        returns
    )

    years = (
        periods
        * HOLDING_DAYS
        / 252.0
    )

    total_return = (
        (1.0 + returns).prod()
        - 1.0
    )

    annual_return = (
        (1.0 + total_return)
        ** (1.0 / years)
        - 1.0
        if years > 0
        else np.nan
    )

    annual_volatility = (
        returns.std(ddof=1)
        * np.sqrt(
            252.0
            / HOLDING_DAYS
        )
        if len(returns) > 1
        else np.nan
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(
            252.0
            / HOLDING_DAYS
        )
        if (
            len(returns) > 1
            and returns.std(ddof=1) > 0
        )
        else np.nan
    )

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

    return {
        "periods":
            periods,

        "total_return":
            total_return,

        "annual_return":
            annual_return,

        "annual_volatility":
            annual_volatility,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "win_rate":
            (returns > 0).mean(),

        "mean_gross":
            gross_returns.mean(),

        "mean_net":
            returns.mean(),

        "mean_turnover":
            df["turnover"].mean(),

        "mean_exposure":
            df["exposure"].mean(),

        "reduced_ratio":
            (
                df["exposure"] < 0.999999
            ).mean(),

        "max_position":
            df["max_position"].max(),

        "mean_beta":
            df["portfolio_beta"].mean(),
    }


# ============================================================
# Summary
# ============================================================

summary_rows = []

for strategy in strategies:

    sdf = result_df[
        result_df["strategy"]
        == strategy
    ]

    periods_to_test = [
        (
            "FULL",
            sdf,
        ),
        (
            "2025",
            sdf[
                (
                    sdf[
                        "prediction_date"
                    ]
                    >= pd.Timestamp(
                        "2025-01-01"
                    )
                )
                & (
                    sdf[
                        "prediction_date"
                    ]
                    <= pd.Timestamp(
                        "2025-12-31"
                    )
                )
            ],
        ),
        (
            "2026",
            sdf[
                (
                    sdf[
                        "prediction_date"
                    ]
                    >= pd.Timestamp(
                        "2026-01-01"
                    )
                )
                & (
                    sdf[
                        "prediction_date"
                    ]
                    <= TEST_END
                )
            ],
        ),
    ]

    for label, subset in (
        periods_to_test
    ):

        row = performance(
            subset
        )

        row["strategy"] = (
            strategy
        )

        row["period"] = (
            label
        )

        summary_rows.append(
            row
        )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Print Summary
# ============================================================

print()
print("=" * 72)
print(
    "V3 TWO-STAGE REGIME RESULTS"
)
print("=" * 72)

print(
    summary_df[
        [
            "strategy",
            "period",
            "periods",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "mean_net",
            "mean_turnover",
            "mean_exposure",
            "reduced_ratio",
            "max_position",
            "mean_beta",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# 2026 Worst Periods
# ============================================================

print()
print("=" * 72)
print(
    "2026 WORST PERIODS"
)
print("=" * 72)

for strategy in strategies:

    print()
    print(
        f"[{strategy}]"
    )

    cols = [
        "prediction_date",
        "entry_date",
        "exit_date",
        "gross_return",
        "net_return",
        "exposure",
        "stress_regime",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "breadth_ma60",
    ]

    worst = (
        result_df[
            result_df["strategy"]
            == strategy
        ]
        .sort_values(
            "net_return"
        )
        .head(10)
    )

    print(
        worst[cols].to_string(
            index=False
        )
    )


# ============================================================
# Save DuckDB
# ============================================================

con = duckdb.connect(
    DB_PATH
)

con.execute(
    f"DROP TABLE IF EXISTS "
    f"{OUTPUT_TABLE}"
)

con.register(
    "result_df_view",
    result_df,
)

con.execute(
    f"""
    CREATE TABLE
    {OUTPUT_TABLE}
    AS
    SELECT *
    FROM result_df_view
    """
)

con.unregister(
    "result_df_view"
)

con.close()


# ============================================================
# Save CSV
# ============================================================

summary_df.to_csv(
    "data/risk_v3_regime_summary.csv",
    index=False,
)

result_df.to_csv(
    "data/risk_v3_regime_detail.csv",
    index=False,
)


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    f"DuckDB table : "
    f"{OUTPUT_TABLE}"
)

print(
    "CSV : "
    "data/risk_v3_regime_summary.csv"
)

print(
    "CSV : "
    "data/risk_v3_regime_detail.csv"
)
