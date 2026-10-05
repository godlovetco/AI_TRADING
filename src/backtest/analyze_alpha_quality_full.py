#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market.duckdb"
OUT_PATH = ROOT / "data" / "alpha_quality_lagged_full.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-09-30")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ============================================================
    # 1. V2 completed trades
    # ============================================================
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
        raise RuntimeError("No V2 validation trades found.")

    for c in ["prediction_date", "entry_date", "exit_date"]:
        v2[c] = pd.to_datetime(v2[c])

    # ============================================================
    # 2. Alpha predictions
    # ============================================================
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
            "Could not identify alpha columns.\n"
            f"columns={cols}\n"
            f"date={date_col}, prediction={pred_col}, target={target_col}"
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

    # ============================================================
    # 3. Daily Alpha Quality
    # ============================================================
    rows = []

    for date, g in alpha.groupby("prediction_date"):

        if len(g) < 5:
            continue

        pred_rank = g["prediction"].rank(
            method="average"
        )

        target_rank = g["target"].rank(
            method="average"
        )

        rank_ic = pred_rank.corr(target_rank)

        n = len(g)
        k = max(1, int(n * 0.20))

        ordered = g.sort_values(
            "prediction"
        )

        bottom = ordered.head(k)["target"].mean()
        top = ordered.tail(k)["target"].mean()

        rows.append(
            {
                "prediction_date": date,
                "rank_ic": rank_ic,
                "top20_return": top,
                "bottom20_return": bottom,
                "top_bottom_spread": top - bottom,
                "alpha_n": n,
            }
        )

    daily = pd.DataFrame(rows)

    if daily.empty:
        raise RuntimeError("Could not calculate daily Alpha statistics.")

    daily = daily.sort_values(
        "prediction_date"
    ).reset_index(drop=True)

    # ============================================================
    # 4. LAGGED Alpha Quality
    #
    # IMPORTANT:
    # All rolling statistics are shifted by one prediction date.
    #
    # Therefore the Alpha Quality for trade T uses only
    # information available BEFORE T.
    # ============================================================

    daily["rank_ic_mean_5"] = (
        daily["rank_ic"]
        .rolling(5, min_periods=5)
        .mean()
        .shift(1)
    )

    daily["rank_ic_mean_10"] = (
        daily["rank_ic"]
        .rolling(10, min_periods=10)
        .mean()
        .shift(1)
    )

    daily["rank_ic_mean_20"] = (
        daily["rank_ic"]
        .rolling(20, min_periods=20)
        .mean()
        .shift(1)
    )

    daily["spread_mean_5"] = (
        daily["top_bottom_spread"]
        .rolling(5, min_periods=5)
        .mean()
        .shift(1)
    )

    daily["spread_mean_10"] = (
        daily["top_bottom_spread"]
        .rolling(10, min_periods=10)
        .mean()
        .shift(1)
    )

    daily["spread_mean_20"] = (
        daily["top_bottom_spread"]
        .rolling(20, min_periods=20)
        .mean()
        .shift(1)
    )

    daily["ic_std_20"] = (
        daily["rank_ic"]
        .rolling(20, min_periods=20)
        .std()
        .shift(1)
    )

    daily["ic_positive_ratio_10"] = (
        daily["rank_ic"]
        .gt(0)
        .rolling(10, min_periods=10)
        .mean()
        .shift(1)
    )

    daily["ic_positive_ratio_20"] = (
        daily["rank_ic"]
        .gt(0)
        .rolling(20, min_periods=20)
        .mean()
        .shift(1)
    )

    # ============================================================
    # 5. Alpha Quality Score
    #
    # This is deliberately simple.
    # No threshold optimization yet.
    #
    # Score components:
    #   + recent IC
    #   + recent spread
    #   + positive IC ratio
    #
    # Standardization is done using expanding historical statistics
    # and then shifted, so no future information leaks backward.
    # ============================================================

    def expanding_z(series):
        mean = (
            series
            .expanding(min_periods=20)
            .mean()
            .shift(1)
        )

        std = (
            series
            .expanding(min_periods=20)
            .std()
            .shift(1)
        )

        return (series - mean) / std.replace(0, np.nan)

    daily["z_ic"] = expanding_z(
        daily["rank_ic_mean_10"]
    )

    daily["z_spread"] = expanding_z(
        daily["spread_mean_10"]
    )

    daily["z_positive_ratio"] = (
        daily["ic_positive_ratio_10"] - 0.5
    )

    daily["alpha_quality_score"] = (
        daily["z_ic"].clip(-3, 3)
        + daily["z_spread"].clip(-3, 3)
        + daily["z_positive_ratio"].clip(-0.5, 0.5)
    )

    # ============================================================
    # 6. Merge with V2 trades
    # ============================================================
    df = v2.merge(
        daily,
        on="prediction_date",
        how="left",
    )

    # Only rows where enough historical Alpha information existed.
    df = df.dropna(
        subset=[
            "rank_ic_mean_10",
            "spread_mean_10",
            "alpha_quality_score",
        ]
    ).copy()

    # ============================================================
    # 7. Quality buckets
    #
    # These are descriptive quantiles only.
    # They are NOT used to choose a strategy threshold.
    # ============================================================
    df["quality_bucket"] = pd.qcut(
        df["alpha_quality_score"],
        q=4,
        labels=[
            "Q1_WEAKEST",
            "Q2",
            "Q3",
            "Q4_STRONGEST",
        ],
        duplicates="drop",
    )

    # ============================================================
    # 8. Results
    # ============================================================
    print()
    print("=" * 90)
    print("LAGGED ALPHA QUALITY VALIDATION")
    print("=" * 90)

    print(f"Period              : {args.start} ~ {args.end}")
    print(f"V2 original trades  : {len(v2)}")
    print(f"Usable lagged trades: {len(df)}")

    print()
    print("-" * 90)
    print("QUALITY BUCKET PERFORMANCE")
    print("-" * 90)

    summary = (
        df.groupby("quality_bucket", observed=True)
        .agg(
            n=("net_return", "size"),
            mean_net=("net_return", "mean"),
            median_net=("net_return", "median"),
            win_rate=(
                "net_return",
                lambda x: (x > 0).mean(),
            ),
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("top_bottom_spread", "mean"),
            mean_lagged_ic=("rank_ic_mean_10", "mean"),
            mean_lagged_spread=(
                "spread_mean_10",
                "mean",
            ),
            mean_quality=(
                "alpha_quality_score",
                "mean",
            ),
        )
        .sort_values("mean_quality")
    )

    print(summary.to_string())

    print()
    print("-" * 90)
    print("CORRELATION: LAGGED QUALITY VS NEXT TRADE")
    print("-" * 90)

    corr_cols = [
        "net_return",
        "gross_return",
        "rank_ic_mean_5",
        "rank_ic_mean_10",
        "rank_ic_mean_20",
        "spread_mean_5",
        "spread_mean_10",
        "spread_mean_20",
        "ic_std_20",
        "ic_positive_ratio_10",
        "ic_positive_ratio_20",
        "alpha_quality_score",
    ]

    print(
        df[corr_cols]
        .corr()["net_return"]
        .sort_values()
        .to_string()
    )

    print()
    print("-" * 90)
    print("WEAK ALPHA EVENTS")
    print("-" * 90)

    weak = df.sort_values(
        "alpha_quality_score"
    ).head(15)

    print(
        weak[
            [
                "prediction_date",
                "net_return",
                "gross_return",
                "rank_ic_mean_10",
                "spread_mean_10",
                "ic_positive_ratio_10",
                "alpha_quality_score",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 90)
    print("STRONG ALPHA EVENTS")
    print("-" * 90)

    strong = df.sort_values(
        "alpha_quality_score",
        ascending=False,
    ).head(15)

    print(
        strong[
            [
                "prediction_date",
                "net_return",
                "gross_return",
                "rank_ic_mean_10",
                "spread_mean_10",
                "ic_positive_ratio_10",
                "alpha_quality_score",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 90)
    print("LOSS RATE BY QUALITY BUCKET")
    print("-" * 90)

    loss_rate = (
        df.groupby("quality_bucket", observed=True)
        .agg(
            n=("net_return", "size"),
            loss_rate=(
                "net_return",
                lambda x: (x < 0).mean(),
            ),
            severe_loss_rate=(
                "net_return",
                lambda x: (x <= -0.03).mean(),
            ),
            mean_net=("net_return", "mean"),
        )
    )

    print(loss_rate.to_string())

    # ============================================================
    # 9. Save
    # ============================================================
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
