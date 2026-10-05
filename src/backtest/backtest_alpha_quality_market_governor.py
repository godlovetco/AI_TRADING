#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market.duckdb"
QUALITY_PATH = ROOT / "data" / "alpha_quality_lagged_full.csv"

TOP_PCT = 0.20
VOL_LOOKBACK = 20
BETA_LOOKBACK = 60

MAX_POSITION = 0.04
MAX_TURNOVER = 0.30
TRANSACTION_COST = 0.001

WEAK_QUANTILE = 0.25


def find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def load_table_schema(con, table):
    return con.execute(
        f"DESCRIBE {table}"
    ).df()["column_name"].tolist()


def prepare_market(con, start, end):

    price_cols = load_table_schema(
        con,
        "daily_prices"
    )

    symbol_col = find_col(
        price_cols,
        ["symbol", "ticker", "code"]
    )

    date_col = find_col(
        price_cols,
        ["date", "trading_date"]
    )

    open_col = find_col(
        price_cols,
        ["open", "open_price"]
    )

    close_col = find_col(
        price_cols,
        ["close", "close_price"]
    )

    if not all(
        [
            symbol_col,
            date_col,
            open_col,
            close_col,
        ]
    ):
        raise RuntimeError(
            f"daily_prices columns={price_cols}"
        )

    prices = con.execute(
        f"""
        SELECT
            "{symbol_col}" AS symbol,
            "{date_col}" AS date,
            "{open_col}" AS open,
            "{close_col}" AS close
        FROM daily_prices
        WHERE "{date_col}" <= ?
        ORDER BY "{symbol_col}", "{date_col}"
        """,
        [end],
    ).df()

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    prices["open"] = pd.to_numeric(
        prices["open"],
        errors="coerce"
    )

    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce"
    )

    prices = prices.dropna(
        subset=[
            "symbol",
            "date",
            "open",
            "close",
        ]
    )

    prices["ret_1d"] = (
        prices
        .groupby("symbol")["close"]
        .pct_change()
    )

    prices["vol_20d"] = (
        prices
        .groupby("symbol")["ret_1d"]
        .rolling(
            VOL_LOOKBACK,
            min_periods=VOL_LOOKBACK,
        )
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)
    )

    return prices


def load_kospi(con, end):

    cols = load_table_schema(
        con,
        "kospi_daily"
    )

    date_col = find_col(
        cols,
        ["date", "trading_date"]
    )

    close_col = find_col(
        cols,
        ["close", "close_price"]
    )

    if not date_col or not close_col:
        raise RuntimeError(
            f"kospi_daily columns={cols}"
        )

    kospi = con.execute(
        f"""
        SELECT
            "{date_col}" AS date,
            "{close_col}" AS close
        FROM kospi_daily
        WHERE "{date_col}" <= ?
        ORDER BY "{date_col}"
        """,
        [end],
    ).df()

    kospi["date"] = pd.to_datetime(
        kospi["date"]
    )

    kospi["close"] = pd.to_numeric(
        kospi["close"],
        errors="coerce"
    )

    kospi = kospi.dropna(
        subset=["date", "close"]
    )

    kospi["kospi_ret"] = (
        kospi["close"].pct_change()
    )

    return kospi


def load_predictions(con, start, end):

    cols = load_table_schema(
        con,
        "alpha_lgbm_predictions"
    )

    date_col = find_col(
        cols,
        ["prediction_date", "date"]
    )

    symbol_col = find_col(
        cols,
        ["symbol", "ticker", "code"]
    )

    pred_col = find_col(
        cols,
        [
            "prediction",
            "predicted",
            "pred",
            "prediction_value",
        ],
    )

    if not all(
        [
            date_col,
            symbol_col,
            pred_col,
        ]
    ):
        raise RuntimeError(
            f"alpha_lgbm_predictions columns={cols}"
        )

    pred = con.execute(
        f"""
        SELECT
            "{date_col}" AS prediction_date,
            "{symbol_col}" AS symbol,
            "{pred_col}" AS prediction
        FROM alpha_lgbm_predictions
        WHERE "{date_col}" >= ?
          AND "{date_col}" <= ?
        ORDER BY "{date_col}", "{symbol_col}"
        """,
        [start, end],
    ).df()

    pred["prediction_date"] = pd.to_datetime(
        pred["prediction_date"]
    )

    pred["prediction"] = pd.to_numeric(
        pred["prediction"],
        errors="coerce"
    )

    pred = pred.dropna(
        subset=[
            "prediction_date",
            "symbol",
            "prediction",
        ]
    )

    return pred


def inverse_vol_cap(g):

    g = g.copy()

    g = g[
        np.isfinite(g["vol_20d"])
        & (g["vol_20d"] > 0)
    ].copy()

    if g.empty:
        return pd.Series(dtype=float)

    raw = 1.0 / g["vol_20d"]

    weights = raw / raw.sum()

    # Iterative 4% cap.
    weights = weights.copy()

    for _ in range(100):

        over = weights > MAX_POSITION

        if not over.any():
            break

        excess = (
            weights[over] - MAX_POSITION
        ).sum()

        weights[over] = MAX_POSITION

        under = ~over

        if not under.any():
            break

        available = weights[under].sum()

        if available <= 0:
            break

        weights[under] += (
            weights[under]
            / available
            * excess
        )

    weights = weights / weights.sum()

    return pd.Series(
        weights.values,
        index=g["symbol"].values,
    )


def calculate_beta_adjustment(
    weights,
    beta,
):

    w = weights.copy()

    valid = (
        beta.reindex(w.index)
        .replace([np.inf, -np.inf], np.nan)
        .notna()
    )

    if not valid.all():
        w = w[valid]
        beta = beta.reindex(w.index)

    if w.empty:
        return w, False, np.nan

    w = w / w.sum()

    portfolio_beta = (
        w * beta
    ).sum()

    # Target beta is 1.0.
    # Only reduce beta if it is above target.
    if portfolio_beta <= 1.01:
        return (
            w,
            False,
            portfolio_beta,
        )

    adjusted = w.copy()

    for _ in range(100):

        portfolio_beta = (
            adjusted * beta
        ).sum()

        if portfolio_beta <= 1.01:
            break

        high = (
            adjusted
            .sort_values(ascending=False)
            .index
        )

        low = (
            beta
            .sort_values()
            .index
        )

        moved = False

        for high_symbol in high:

            if adjusted[high_symbol] <= 0:
                continue

            for low_symbol in low:

                if (
                    low_symbol
                    == high_symbol
                ):
                    continue

                if (
                    beta[low_symbol]
                    >= beta[high_symbol]
                ):
                    continue

                room = (
                    MAX_POSITION
                    - adjusted[low_symbol]
                )

                if room <= 1e-12:
                    continue

                delta = min(
                    adjusted[high_symbol],
                    room,
                    0.01,
                )

                adjusted[
                    high_symbol
                ] -= delta

                adjusted[
                    low_symbol
                ] += delta

                moved = True
                break

            if moved:
                break

        if not moved:
            break

    adjusted = adjusted / adjusted.sum()

    final_beta = (
        adjusted * beta
    ).sum()

    return (
        adjusted,
        True,
        final_beta,
    )


def constrain_turnover(
    target,
    previous,
):

    if previous is None:

        return target.copy(), 1.0

    idx = (
        target.index
        .union(previous.index)
    )

    t = target.reindex(
        idx,
        fill_value=0.0
    )

    p = previous.reindex(
        idx,
        fill_value=0.0
    )

    raw_turnover = (
        np.abs(t - p).sum()
        / 2
    )

    if raw_turnover <= MAX_TURNOVER:

        constrained = t

    else:

        scale = (
            MAX_TURNOVER
            / raw_turnover
        )

        constrained = (
            p
            + scale * (t - p)
        )

    constrained = constrained[
        constrained > 1e-12
    ]

    return constrained, min(
        raw_turnover,
        MAX_TURNOVER,
    )


def compute_trade_returns(
    prices,
    entry_date,
    exit_date,
    weights,
):

    available = prices[
        prices["symbol"].isin(
            weights.index
        )
    ]

    entry = available[
        available["date"]
        == entry_date
    ][
        ["symbol", "open"]
    ].rename(
        columns={
            "open": "entry_open"
        }
    )

    exit_ = available[
        available["date"]
        == exit_date
    ][
        ["symbol", "open"]
    ].rename(
        columns={
            "open": "exit_open"
        }
    )

    px = entry.merge(
        exit_,
        on="symbol",
        how="inner",
    )

    if px.empty:
        return 0.0, 0.0, len(weights)

    px["return"] = (
        px["exit_open"]
        / px["entry_open"]
        - 1
    )

    w = weights.reindex(
        px["symbol"]
    ).fillna(0)

    gross = (
        w.values
        * px["return"].values
    ).sum()

    missing_weight = (
        weights.sum()
        - w.sum()
    )

    # Missing stocks are treated as cash.
    return (
        gross,
        missing_weight,
        len(px),
    )


def performance_metrics(df):

    r = df["net_return"].values

    equity = np.cumprod(
        1 + r
    )

    total = equity[-1] - 1

    years = (
        len(r) * 5 / 252
    )

    annual = (
        (1 + total) ** (1 / years)
        - 1
    )

    vol = (
        np.std(r, ddof=1)
        * np.sqrt(252 / 5)
    )

    sharpe = (
        np.mean(r)
        / np.std(r, ddof=1)
        * np.sqrt(252 / 5)
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = equity / peak - 1

    return {
        "periods": len(r),
        "total_return": total,
        "annual_return": annual,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "win_rate": np.mean(r > 0),
        "mean_net": np.mean(r),
        "median_net": np.median(r),
        "severe_loss_rate": np.mean(
            r <= -0.03
        ),
        "worst_trade": r.min(),
        "best_trade": r.max(),
        "mean_exposure": df[
            "exposure"
        ].mean(),
        "mean_turnover": df[
            "turnover"
        ].mean(),
        "total_transaction_cost": df[
            "transaction_cost"
        ].sum(),
    }



# ------------------------------------------------------------
# Market Risk Governor
# ------------------------------------------------------------

def build_market_governor(kospi, prices, calibration_end):

    k = kospi.copy()

    # KOSPI trend
    k["ma20"] = (
        k["close"]
        .rolling(20)
        .mean()
    )

    k["ma60"] = (
        k["close"]
        .rolling(60)
        .mean()
    )

    # --------------------------------------------------------
    # Stock-level breadth.
    #
    # Breadth = proportion of stocks whose close is above
    # their own 60D moving average on each date.
    #
    # This is the same market breadth concept used in the
    # previously validated regime diagnostics.
    # --------------------------------------------------------

    stock = prices[
        [
            "symbol",
            "date",
            "close",
        ]
    ].copy()

    stock = stock.sort_values(
        ["symbol", "date"]
    )

    stock["ma60"] = (
        stock
        .groupby("symbol")["close"]
        .transform(
            lambda x: x.rolling(
                60,
                min_periods=60,
            ).mean()
        )
    )

    stock["above_ma60"] = (
        stock["close"]
        > stock["ma60"]
    )

    breadth = (
        stock
        .dropna(subset=["ma60"])
        .groupby("date")["above_ma60"]
        .mean()
        .rename("breadth_ma60")
        .reset_index()
    )

    k = k.merge(
        breadth,
        on="date",
        how="left",
    )

    # 60D drawdown
    k["drawdown60"] = (
        k["close"]
        / k["close"].rolling(60).max()
        - 1
    )

    # Volatility
    k["vol20"] = (
        k["kospi_ret"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    k["vol60"] = (
        k["kospi_ret"]
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    k["vol_ratio_20_60"] = (
        k["vol20"] / k["vol60"]
    )

    # --------------------------------------------------------
    # Calibration ONLY through calibration_end.
    # --------------------------------------------------------

    cal = k[
        k["date"]
        <= pd.Timestamp(calibration_end)
    ].copy()

    thresholds = {
        "breadth_q20": cal[
            "breadth_ma60"
        ].quantile(0.20),

        "breadth_q10": cal[
            "breadth_ma60"
        ].quantile(0.10),

        "drawdown_q20": cal[
            "drawdown60"
        ].quantile(0.20),

        "vol_q80": cal[
            "vol20"
        ].quantile(0.80),

        "vol_q90": cal[
            "vol20"
        ].quantile(0.90),

        "vol_q95": cal[
            "vol20"
        ].quantile(0.95),
    }

    return k, thresholds


def calculate_market_exposure(
    row,
    thresholds,
    candidate,
):

    score = 0

    breadth = row["breadth_ma60"]
    close = row["close"]
    ma20 = row["ma20"]
    ma60 = row["ma60"]
    drawdown = row["drawdown60"]
    vol = row["vol20"]

    # --------------------------------------------------------
    # Breadth
    # --------------------------------------------------------

    if (
        np.isfinite(breadth)
        and breadth <= thresholds["breadth_q20"]
    ):
        score += 1

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if (
        np.isfinite(close)
        and np.isfinite(ma20)
        and close < ma20
    ):
        score += 1

    if (
        np.isfinite(close)
        and np.isfinite(ma60)
        and close < ma60
    ):
        score += 1

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    if (
        np.isfinite(drawdown)
        and drawdown <= thresholds["drawdown_q20"]
    ):
        score += 1

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    if candidate == "C1":

        if (
            np.isfinite(vol)
            and vol >= thresholds["vol_q80"]
        ):
            score += 1

    elif candidate == "C2":

        if (
            np.isfinite(vol)
            and vol >= thresholds["vol_q80"]
        ):
            score += 1

        # Additional volatility severity signal.
        if (
            np.isfinite(vol)
            and vol >= thresholds["vol_q90"]
        ):
            score += 1

    else:

        raise ValueError(
            f"Unknown market governor candidate: {candidate}"
        )

    # --------------------------------------------------------
    # Exposure mapping
    # --------------------------------------------------------

    if score >= 4:
        exposure = 0.25

    elif score >= 3:
        exposure = 0.50

    elif score >= 2:
        exposure = 0.75

    else:
        exposure = 1.00

    return exposure, score

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--end",
        default="2025-12-31",
    )

    parser.add_argument(
        "--calibration-end",
        default="2024-12-31",
    )

    args = parser.parse_args()

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    prices = prepare_market(
        con,
        args.start,
        args.end,
    )

    kospi = load_kospi(
        con,
        args.end,
    )

    kospi, market_thresholds = build_market_governor(
        kospi,
        prices,
        args.calibration_end,
    )

    pred = load_predictions(
        con,
        args.start,
        args.end,
    )

    con.close()

    # ------------------------------------------------------------
    # Build 60D rolling beta against KOSPI.
    # ------------------------------------------------------------

    m = prices.merge(
        kospi[
            ["date", "kospi_ret"]
        ],
        on="date",
        how="left",
    )

    def beta_calc(g):

        market = g["kospi_ret"]
        stock = g["ret_1d"]

        cov = (
            stock.rolling(
                BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            )
            .cov(market)
        )

        var = (
            market.rolling(
                BETA_LOOKBACK,
                min_periods=BETA_LOOKBACK,
            )
            .var()
        )

        return cov / var

    m["beta_60d"] = (
        m.groupby("symbol", group_keys=False)
        .apply(
            beta_calc,
            include_groups=False,
        )
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # ------------------------------------------------------------
    # Alpha Quality.
    # ------------------------------------------------------------

    quality = pd.read_csv(
        QUALITY_PATH
    )

    quality["prediction_date"] = pd.to_datetime(
        quality["prediction_date"]
    )

    quality = quality[
        [
            "prediction_date",
            "alpha_quality_score",
        ]
    ].drop_duplicates(
        "prediction_date"
    )

    # ------------------------------------------------------------
    # Prediction dates.
    # Same 5-date non-overlap convention as V2.
    # ------------------------------------------------------------

    dates = (
        pred["prediction_date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Every 5th prediction date.
    rebalance_dates = dates[::5]

    # Need a later prediction/trading date.
    completed_dates = []

    for i in range(
        len(rebalance_dates) - 1
    ):

        d = rebalance_dates[i]

        future_dates = [
            x for x in dates
            if x > d
        ]

        if not future_dates:
            continue

        completed_dates.append(d)

    # ------------------------------------------------------------
    # Calibration threshold.
    # IMPORTANT:
    # ONLY 2024 is used.
    # ------------------------------------------------------------

    # IMPORTANT:
    # The weak threshold was calibrated previously on the
    # 2024 V2 validation-trade sample.
    #
    # Fixed threshold:
    #     -1.519327
    #
    # Do NOT recalculate it from the expanded full daily
    # Alpha Quality series. That would change the calibration
    # population and contaminate the OOS/final comparison.

    weak_threshold = -1.519327

    print()
    print("=" * 100)
    print("STOCK-LEVEL V2 + ALPHA QUALITY GOVERNOR")
    print("=" * 100)

    print(
        f"Period              : "
        f"{args.start} ~ {args.end}"
    )

    print(
        f"Calibration end     : "
        f"{args.calibration_end}"
    )

    print(
        f"Fixed weak threshold: "
        f"{weak_threshold:.6f}"
    )

    print(
        f"Prediction dates    : "
        f"{len(dates)}"
    )

    print(
        f"Completed periods   : "
        f"{len(completed_dates)}"
    )

    print()
    print("-" * 100)
    print("MARKET GOVERNOR CALIBRATION THRESHOLDS")
    print("-" * 100)

    for key, value in market_thresholds.items():
        print(
            f"{key:20s}: {value:.6f}"
        )

    # ------------------------------------------------------------
    # Strategy states.
    # ------------------------------------------------------------

    strategies = {
        "V2_BASELINE": {
            "aqg_exposure": 1.00,
            "market_governor": None,
        },
        "AQG_75": {
            "aqg_exposure": 0.75,
            "market_governor": None,
        },
        "AQG_50": {
            "aqg_exposure": 0.50,
            "market_governor": None,
        },
        "AQG75_C1": {
            "aqg_exposure": 0.75,
            "market_governor": "C1",
        },
        "AQG75_C2": {
            "aqg_exposure": 0.75,
            "market_governor": "C2",
        },
    }

    all_trades = []
    all_weights = []

    previous_weights = {
        name: None
        for name in strategies
    }

    for period_idx, prediction_date in enumerate(
        completed_dates
    ):

        future_dates = [
            x for x in dates
            if x > prediction_date
        ]

        if len(future_dates) < 2:
            continue

        entry_date = future_dates[0]

        # Five trading-day open-to-open convention.
        # Find the 5th trading date after entry.
        exit_candidates = [
            x for x in dates
            if x > entry_date
        ]

        if len(exit_candidates) < 5:
            continue

        exit_date = exit_candidates[4]

        p = pred[
            pred["prediction_date"]
            == prediction_date
        ].copy()

        if p.empty:
            continue

        # Vol/beta snapshot at prediction date.
        risk = m[
            m["date"]
            == prediction_date
        ][
            [
                "symbol",
                "vol_20d",
                "beta_60d",
            ]
        ]

        p = p.merge(
            risk,
            on="symbol",
            how="inner",
        )

        p = p[
            np.isfinite(p["prediction"])
            & np.isfinite(p["vol_20d"])
            & np.isfinite(p["beta_60d"])
            & (p["vol_20d"] > 0)
        ].copy()

        if p.empty:
            continue

        # --------------------------------------------------------
        # Top 20%
        # --------------------------------------------------------

        n_selected = max(
            1,
            int(
                len(p) * TOP_PCT
            ),
        )

        p = p.sort_values(
            "prediction",
            ascending=False,
        ).head(
            n_selected
        )

        base_weights = inverse_vol_cap(
            p[
                [
                    "symbol",
                    "vol_20d",
                ]
            ]
        )

        if base_weights.empty:
            continue

        beta = (
            p.set_index("symbol")
            ["beta_60d"]
        )

        base_weights = (
            base_weights
            .reindex(beta.index)
            .dropna()
        )

        beta = beta.reindex(
            base_weights.index
        )

        base_weights = (
            base_weights
            / base_weights.sum()
        )

        base_weights, beta_adjusted, portfolio_beta = (
            calculate_beta_adjustment(
                base_weights,
                beta,
            )
        )

        # --------------------------------------------------------
        # Market Governor snapshot at prediction date.
        #
        # Strictly use information available on prediction_date.
        # --------------------------------------------------------

        market_row = kospi[
            kospi["date"]
            == prediction_date
        ]

        if market_row.empty:
            continue

        current_market = market_row.iloc[0]

        quality_row = quality[
            quality["prediction_date"]
            == prediction_date
        ]

        if quality_row.empty:
            alpha_quality = np.nan
        else:
            alpha_quality = float(
                quality_row[
                    "alpha_quality_score"
                ].iloc[0]
            )

        weak_alpha = (
            np.isfinite(alpha_quality)
            and alpha_quality
            <= weak_threshold
        )

        # --------------------------------------------------------
        # Each strategy.
        # --------------------------------------------------------

        for strategy, config in strategies.items():

            # ----------------------------------------------------
            # 1. Alpha Quality Governor
            # ----------------------------------------------------

            normal_exposure = config["aqg_exposure"]

            if strategy == "V2_BASELINE":

                aqg_exposure = 1.0

            elif weak_alpha:

                aqg_exposure = normal_exposure

            else:

                aqg_exposure = 1.0

            # ----------------------------------------------------
            # 2. Market Governor
            #
            # Applied AFTER Alpha Quality Governor.
            # ----------------------------------------------------

            market_candidate = config[
                "market_governor"
            ]

            if market_candidate is not None:

                market_exposure, market_score = (
                    calculate_market_exposure(
                        current_market,
                        market_thresholds,
                        market_candidate,
                    )
                )

            else:

                market_exposure = 1.0
                market_score = 0

            # ----------------------------------------------------
            # 3. Combined exposure
            # ----------------------------------------------------

            exposure = (
                aqg_exposure
                * market_exposure
            )

            target_stock = (
                base_weights
                * exposure
            )

            target_stock = (
                target_stock[
                    target_stock > 1e-12
                ]
            )

            constrained, raw_turnover = (
                constrain_turnover(
                    target_stock,
                    previous_weights[
                        strategy
                    ],
                )
            )

            # Explicit cash.
            stock_sum = constrained.sum()

            cash = max(
                0.0,
                1.0 - stock_sum,
            )

            final_weights = constrained.copy()

            if cash > 1e-12:
                final_weights[
                    "__CASH__"
                ] = cash

            # Actual turnover is based on total portfolio weights.
            previous = previous_weights[
                strategy
            ]

            if previous is None:

                actual_turnover = 1.0

            else:

                idx = (
                    final_weights.index
                    .union(
                        previous.index
                    )
                )

                a = final_weights.reindex(
                    idx,
                    fill_value=0.0,
                )

                b = previous.reindex(
                    idx,
                    fill_value=0.0,
                )

                actual_turnover = (
                    np.abs(a - b).sum()
                    / 2
                )

            # ----------------------------------------------------
            # Open-to-open stock returns.
            # ----------------------------------------------------

            stock_weights = final_weights[
                final_weights.index
                != "__CASH__"
            ]

            gross, missing_weight, n_available = (
                compute_trade_returns(
                    prices=m,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    weights=stock_weights,
                )
            )

            # Missing stock weight remains cash.
            cash_weight = (
                final_weights.get(
                    "__CASH__",
                    0.0,
                )
                + missing_weight
            )

            transaction_cost = (
                actual_turnover
                * TRANSACTION_COST
            )

            net = (
                gross
                - transaction_cost
            )

            all_trades.append(
                {
                    "strategy": strategy,
                    "period": period_idx,
                    "prediction_date": prediction_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "n_selected": len(
                        stock_weights
                    ),
                    "gross_return": gross,
                    "turnover": actual_turnover,
                    "transaction_cost": transaction_cost,
                    "net_return": net,
                    "exposure": stock_weights.sum(),
                    "cash_weight": cash_weight,
                    "weak_alpha": weak_alpha,
                    "alpha_quality_score": alpha_quality,
                    "weak_threshold": weak_threshold,
                    "portfolio_beta": portfolio_beta,
                    "beta_adjusted": beta_adjusted,
                    "n_price_available": n_available,
                    "max_position": (
                        stock_weights.max()
                        if len(stock_weights)
                        else 0.0
                    ),
                                        # Market Governor diagnostics
                    "market_candidate": market_candidate,
                    "market_score": market_score,
                    "market_exposure": market_exposure,
                    "market_breadth_ma60": current_market["breadth_ma60"],
                    "market_close": current_market["close"],
                    "market_ma20": current_market["ma20"],
                    "market_ma60": current_market["ma60"],
                    "market_drawdown60": current_market["drawdown60"],
                    "market_vol20": current_market["vol20"],
                }
            )

            for symbol, weight in final_weights.items():

                all_weights.append(
                    {
                        "strategy": strategy,
                        "period": period_idx,
                        "prediction_date": prediction_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "symbol": symbol,
                        "weight": weight,
                        "weak_alpha": weak_alpha,

                        # Market Governor diagnostics
                        "market_candidate": market_candidate,
                        "market_score": market_score,
                        "market_exposure": market_exposure,
                        "market_breadth_ma60": current_market["breadth_ma60"],
                        "market_close": current_market["close"],
                        "market_ma20": current_market["ma20"],
                        "market_ma60": current_market["ma60"],
                        "market_drawdown60": current_market["drawdown60"],
                        "market_vol20": current_market["vol20"],
                        "alpha_quality_score": alpha_quality,
                    }
                )

            previous_weights[
                strategy
            ] = final_weights

    trades = pd.DataFrame(
        all_trades
    )

    weights = pd.DataFrame(
        all_weights
    )

    if trades.empty:
        raise RuntimeError(
            "No completed trades generated."
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    summary_rows = []

    for strategy, g in trades.groupby(
        "strategy"
    ):

        metrics = performance_metrics(
            g
        )

        metrics[
            "strategy"
        ] = strategy

        summary_rows.append(
            metrics
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("-" * 100)
    print("FULL 2024-2025 RESULTS")
    print("-" * 100)

    print(
        summary[
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
                "severe_loss_rate",
                "worst_trade",
                "mean_exposure",
                "mean_turnover",
                "total_transaction_cost",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # 2024 / 2025 split
    # ------------------------------------------------------------

    print()
    print("-" * 100)
    print("YEARLY RESULTS")
    print("-" * 100)

    yearly = []

    for (
        strategy,
        year,
    ), g in trades.groupby(
        [
            "strategy",
            trades["prediction_date"].dt.year,
        ]
    ):

        metrics = performance_metrics(
            g
        )

        metrics[
            "strategy"
        ] = strategy

        metrics[
            "year"
        ] = year

        yearly.append(
            metrics
        )

    yearly = pd.DataFrame(
        yearly
    )

    print(
        yearly[
            [
                "year",
                "strategy",
                "periods",
                "total_return",
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "win_rate",
                "mean_net",
                "severe_loss_rate",
                "worst_trade",
                "mean_exposure",
                "mean_turnover",
            ]
        ].sort_values(
            [
                "year",
                "strategy",
            ]
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # OOS only
    # ------------------------------------------------------------

    oos = trades[
        trades["prediction_date"]
        >= pd.Timestamp("2025-01-01")
    ].copy()

    print()
    print("-" * 100)
    print("2025 OOS — STOCK LEVEL")
    print("-" * 100)

    oos_rows = []

    for strategy, g in oos.groupby(
        "strategy"
    ):

        metrics = performance_metrics(
            g
        )

        metrics[
            "strategy"
        ] = strategy

        oos_rows.append(
            metrics
        )

    oos_summary = pd.DataFrame(
        oos_rows
    )

    print(
        oos_summary[
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
                "severe_loss_rate",
                "worst_trade",
                "mean_exposure",
                "mean_turnover",
                "total_transaction_cost",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Worst OOS trades
    # ------------------------------------------------------------

    print()
    print("-" * 100)
    print("2025 OOS WORST TRADES")
    print("-" * 100)

    print(
        oos.sort_values(
            "net_return"
        ).groupby(
            "strategy",
            group_keys=False,
        ).head(5)[
            [
                "strategy",
                "prediction_date",
                "net_return",
                "gross_return",
                "turnover",
                "transaction_cost",
                "exposure",
                "cash_weight",
                "weak_alpha",
                "alpha_quality_score",
                "portfolio_beta",
                "max_position",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    summary_path = (
        ROOT
        / "data"
        / "alpha_quality_stocklevel_summary.csv"
    )

    yearly_path = (
        ROOT
        / "data"
        / "alpha_quality_stocklevel_yearly.csv"
    )

    oos_path = (
        ROOT
        / "data"
        / "alpha_quality_stocklevel_oos.csv"
    )

    trades_path = (
        ROOT
        / "data"
        / "alpha_quality_stocklevel_trades.csv"
    )

    weights_path = (
        ROOT
        / "data"
        / "alpha_quality_stocklevel_weights.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    yearly.to_csv(
        yearly_path,
        index=False,
    )

    oos_summary.to_csv(
        oos_path,
        index=False,
    )

    trades.to_csv(
        trades_path,
        index=False,
    )

    weights.to_csv(
        weights_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(summary_path)
    print(yearly_path)
    print(oos_path)
    print(trades_path)
    print(weights_path)


if __name__ == "__main__":
    main()
