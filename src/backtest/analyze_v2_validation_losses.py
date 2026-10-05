#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market.duckdb"
OUT_PATH = ROOT / "data" / "v2_validation_loss_attribution.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ------------------------------------------------------------
    # 1. V2 completed trades
    # ------------------------------------------------------------
    v2 = con.execute(
        """
        SELECT *
        FROM alpha_lgbm_risk_v2_beta_backtest
        WHERE prediction_date >= ?
          AND prediction_date <= ?
        ORDER BY prediction_date
        """,
        [args.start, args.end],
    ).df()

    if v2.empty:
        raise RuntimeError("V2 validation trades not found.")

    for c in ["prediction_date", "entry_date", "exit_date"]:
        v2[c] = pd.to_datetime(v2[c])

    # ------------------------------------------------------------
    # 2. KOSPI regime variables
    # ------------------------------------------------------------
    kospi = con.execute(
        """
        SELECT *
        FROM kospi_daily
        ORDER BY date
        """
    ).df()

    kospi.columns = [c.lower() for c in kospi.columns]

    if "date" not in kospi.columns or "close" not in kospi.columns:
        raise RuntimeError(
            f"kospi_daily columns are: {list(kospi.columns)}"
        )

    kospi["date"] = pd.to_datetime(kospi["date"])
    kospi["close"] = pd.to_numeric(kospi["close"], errors="coerce")
    kospi = kospi.dropna(subset=["date", "close"])
    kospi = kospi.sort_values("date").drop_duplicates("date")

    kospi["ret_1d"] = kospi["close"].pct_change()
    kospi["ret_5d"] = kospi["close"].pct_change(5)
    kospi["ret_20d"] = kospi["close"].pct_change(20)

    kospi["vol_20d"] = (
        kospi["ret_1d"].rolling(20).std() * np.sqrt(252)
    )

    kospi["vol_60d"] = (
        kospi["ret_1d"].rolling(60).std() * np.sqrt(252)
    )

    kospi["vol_ratio_20_60"] = (
        kospi["vol_20d"] / kospi["vol_60d"]
    )

    # Forward market returns for diagnosis.
    kospi["forward_5d"] = (
        kospi["close"].shift(-5) / kospi["close"] - 1
    )

    kospi["forward_20d"] = (
        kospi["close"].shift(-20) / kospi["close"] - 1
    )

    # ------------------------------------------------------------
    # 3. Breadth
    # Existing project convention:
    # fraction of stocks trading above their own 60-day MA.
    # ------------------------------------------------------------
    prices = con.execute(
        """
        SELECT symbol, date, close
        FROM daily_prices
        ORDER BY symbol, date
        """
    ).df()

    prices["date"] = pd.to_datetime(prices["date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    prices = prices.dropna(
        subset=["symbol", "date", "close"]
    )

    prices = prices.drop_duplicates(
        ["symbol", "date"]
    )

    prices["ma60"] = (
        prices.groupby("symbol")["close"]
        .transform(
            lambda x: x.rolling(60, min_periods=60).mean()
        )
    )

    prices["above_ma60"] = (
        prices["close"] > prices["ma60"]
    )

    breadth = (
        prices.groupby("date")["above_ma60"]
        .mean()
        .rename("breadth_ma60")
        .reset_index()
    )

    regime = kospi.merge(
        breadth,
        on="date",
        how="left",
    )

    regime = regime.rename(
        columns={"date": "prediction_date"}
    )

    # ------------------------------------------------------------
    # 4. Existing alpha predictions
    # ------------------------------------------------------------
    cols = con.execute(
        "DESCRIBE alpha_lgbm_predictions"
    ).df()["column_name"].tolist()

    lower = {c.lower(): c for c in cols}

    def find(names):
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    date_col = find(
        ["prediction_date", "date", "datetime"]
    )

    pred_col = find(
        ["prediction", "predicted", "prediction_value", "pred"]
    )

    target_col = find(
        [
            "future_excess_return_5d",
            "target",
            "actual",
            "future_return_5d",
            "actual_return",
        ]
    )

    if not date_col or not pred_col or not target_col:
        raise RuntimeError(
            "Could not identify alpha prediction columns.\n"
            f"Columns = {cols}\n"
            f"date={date_col}, prediction={pred_col}, "
            f"target={target_col}"
        )

    alpha = con.execute(
        f"""
        SELECT
            "{date_col}" AS prediction_date,
            "{pred_col}" AS prediction,
            "{target_col}" AS target
        FROM alpha_lgbm_predictions
        WHERE "{date_col}" >= ?
          AND "{date_col}" <= ?
        """,
        [args.start, args.end],
    ).df()

    alpha["prediction_date"] = pd.to_datetime(
        alpha["prediction_date"]
    )

    alpha["prediction"] = pd.to_numeric(
        alpha["prediction"],
        errors="coerce",
    )

    alpha["target"] = pd.to_numeric(
        alpha["target"],
        errors="coerce",
    )

    alpha = alpha.dropna(
        subset=["prediction", "target"]
    )

    # Daily alpha quality.
    alpha_stats = []

    for date, g in alpha.groupby("prediction_date"):

        if len(g) < 5:
            continue

        rank_ic = (
            g["prediction"]
            .rank()
            .corr(g["target"].rank())
        )

        n = len(g)
        k = max(1, int(n * 0.20))

        ordered = g.sort_values("prediction")

        bottom = ordered.head(k)["target"].mean()
        top = ordered.tail(k)["target"].mean()

        alpha_stats.append(
            {
                "prediction_date": date,
                "rank_ic": rank_ic,
                "top20_return": top,
                "bottom20_return": bottom,
                "top_bottom_spread": top - bottom,
                "alpha_n": n,
            }
        )

    alpha_stats = pd.DataFrame(alpha_stats)

    # ------------------------------------------------------------
    # 5. Merge
    # ------------------------------------------------------------
    df = v2.merge(
        regime,
        on="prediction_date",
        how="left",
    )

    df = df.merge(
        alpha_stats,
        on="prediction_date",
        how="left",
    )

    # ------------------------------------------------------------
    # 6. Actual KOSPI return during each V2 holding period
    # ------------------------------------------------------------
    close_map = kospi.set_index("date")["close"]

    def market_return(row):
        try:
            entry = close_map.loc[row["entry_date"]]
            exit_ = close_map.loc[row["exit_date"]]
            return exit_ / entry - 1
        except KeyError:
            return np.nan

    df["kospi_holding_return"] = df.apply(
        market_return,
        axis=1,
    )

    df["alpha_minus_kospi"] = (
        df["gross_return"]
        - df["kospi_holding_return"]
    )

    # ------------------------------------------------------------
    # 7. Existing regime labels
    # ------------------------------------------------------------
    q80 = df["vol_20d"].quantile(0.80)
    q20 = df["vol_20d"].quantile(0.20)

    df["vol_regime"] = np.select(
        [
            df["vol_20d"] <= q20,
            df["vol_20d"] >= q80,
        ],
        [
            "LOW_VOL",
            "HIGH_VOL",
        ],
        default="NORMAL_VOL",
    )

    df["trend_regime"] = np.select(
        [
            df["ret_20d"] <= -0.10,
            df["ret_20d"] >= 0.10,
        ],
        [
            "BEAR",
            "BULL",
        ],
        default="TRANSITION",
    )

    df["breadth_regime"] = np.select(
        [
            df["breadth_ma60"] <= 0.30,
            df["breadth_ma60"] >= 0.70,
        ],
        [
            "BROAD_BEAR",
            "BROAD_BULL",
        ],
        default="MIXED",
    )

    high_vol = df["vol_20d"] >= q80

    hvr = (
        high_vol
        & (df["ret_20d"] >= 0.10)
        & (df["ret_5d"] <= -0.05)
    )

    downside = (
        high_vol
        & (df["ret_20d"] <= -0.10)
    )

    df["stress_regime"] = np.select(
        [
            downside,
            hvr,
        ],
        [
            "DOWNSIDE_STRESS",
            "HIGH_VOL_RALLY",
        ],
        default="NORMAL",
    )

    # Diagnostic only. Not a new trading rule.
    df["validation_stress_flag"] = (
        (df["vol_20d"] >= 0.30)
        & (df["breadth_ma60"] <= 0.40)
    )

    df["loss_bucket"] = np.select(
        [
            df["net_return"] <= -0.05,
            df["net_return"] <= -0.03,
            df["net_return"] < 0,
        ],
        [
            "SEVERE_LOSS_5P",
            "LOSS_3P_5P",
            "LOSS_LT_3P",
        ],
        default="NON_LOSS",
    )

    # ------------------------------------------------------------
    # 8. Print results
    # ------------------------------------------------------------
    print()
    print("=" * 90)
    print("V2 VALIDATION LOSS ATTRIBUTION")
    print("=" * 90)

    print(f"Period              : {args.start} ~ {args.end}")
    print(f"Completed V2 trades : {len(df)}")
    print(
        f"Mean net return     : "
        f"{df['net_return'].mean():.4%}"
    )
    print(
        f"Win rate            : "
        f"{(df['net_return'] > 0).mean():.2%}"
    )

    print()
    print("-" * 90)
    print("WORST 15 TRADES")
    print("-" * 90)

    show = [
        "prediction_date",
        "entry_date",
        "exit_date",
        "net_return",
        "gross_return",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "breadth_ma60",
        "stress_regime",
        "validation_stress_flag",
        "rank_ic",
        "top_bottom_spread",
        "portfolio_beta",
        "effective_n",
        "HHI",
        "kospi_holding_return",
        "alpha_minus_kospi",
    ]

    print(
        df.sort_values("net_return")[
            show
        ].head(15).to_string(index=False)
    )

    print()
    print("-" * 90)
    print("LOSS BUCKET SUMMARY")
    print("-" * 90)

    bucket = (
        df.groupby("loss_bucket")
        .agg(
            n=("net_return", "size"),
            mean_net=("net_return", "mean"),
            median_net=("net_return", "median"),
            mean_beta=("portfolio_beta", "mean"),
            mean_effective_n=("effective_n", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("top_bottom_spread", "mean"),
            mean_vol=("vol_20d", "mean"),
            mean_breadth=("breadth_ma60", "mean"),
            mean_kospi_hold=("kospi_holding_return", "mean"),
        )
        .sort_values("mean_net")
    )

    print(bucket.to_string())

    print()
    print("-" * 90)
    print("REGIME SUMMARY")
    print("-" * 90)

    for col in [
        "vol_regime",
        "trend_regime",
        "breadth_regime",
        "stress_regime",
    ]:
        print()
        print(f"[{col}]")

        result = (
            df.groupby(col)
            .agg(
                n=("net_return", "size"),
                mean_net=("net_return", "mean"),
                median_net=("net_return", "median"),
                win_rate=(
                    "net_return",
                    lambda x: (x > 0).mean(),
                ),
                mean_rank_ic=("rank_ic", "mean"),
                mean_spread=(
                    "top_bottom_spread",
                    "mean",
                ),
                mean_kospi_hold=(
                    "kospi_holding_return",
                    "mean",
                ),
            )
            .sort_values("mean_net")
        )

        print(result.to_string())

    print()
    print("-" * 90)
    print("SEVERE LOSS COUNTS (NET <= -3%)")
    print("-" * 90)

    severe = df["net_return"] <= -0.03

    for col in [
        "vol_regime",
        "trend_regime",
        "breadth_regime",
        "stress_regime",
        "validation_stress_flag",
    ]:
        print()
        print(f"[{col}]")
        print(
            df.loc[severe, col]
            .value_counts(dropna=False)
            .to_string()
        )

    print()
    print("-" * 90)
    print("CORRELATION WITH NET RETURN")
    print("-" * 90)

    numeric = [
        "net_return",
        "gross_return",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "breadth_ma60",
        "rank_ic",
        "top_bottom_spread",
        "portfolio_beta",
        "effective_n",
        "HHI",
        "kospi_holding_return",
        "alpha_minus_kospi",
    ]

    print(
        df[numeric]
        .corr()["net_return"]
        .sort_values()
        .to_string()
    )

    # ------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------
    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("=" * 90)
    print(f"Saved: {OUT_PATH}")
    print("=" * 90)

    con.close()


if __name__ == "__main__":
    main()
