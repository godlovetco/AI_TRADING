#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market.duckdb"
OUT_PATH = ROOT / "data" / "alpha_quality_lagged_full.csv"


def find_column(columns, names):
    lower = {c.lower(): c for c in columns}

    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]

    return None


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

    return (
        (series - mean)
        / std.replace(0, np.nan)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--end",
        default="2026-09-30",
    )

    args = parser.parse_args()

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    # ============================================================
    # 1. Alpha predictions
    #
    # IMPORTANT:
    # Do NOT use V2 backtest dates here.
    #
    # Alpha Quality must be calculated for every available
    # prediction date so that 2026 can be evaluated.
    # ============================================================

    cols = con.execute(
        "DESCRIBE alpha_lgbm_predictions"
    ).df()["column_name"].tolist()

    date_col = find_column(
        cols,
        [
            "prediction_date",
            "date",
            "datetime",
        ],
    )

    pred_col = find_column(
        cols,
        [
            "prediction",
            "predicted",
            "prediction_value",
            "pred",
        ],
    )

    target_col = find_column(
        cols,
        [
            "future_excess_return_5d",
            "target",
            "actual",
            "future_return_5d",
            "actual_return",
        ],
    )

    if not date_col or not pred_col or not target_col:

        raise RuntimeError(
            "Could not identify alpha columns.\n"
            f"columns={cols}\n"
            f"date={date_col}, "
            f"prediction={pred_col}, "
            f"target={target_col}"
        )

    print()
    print("=" * 100)
    print("BUILD FULL ALPHA QUALITY")
    print("=" * 100)

    print(
        f"Period: "
        f"{args.start} ~ {args.end}"
    )

    print(
        f"Date column   : {date_col}"
    )

    print(
        f"Prediction    : {pred_col}"
    )

    print(
        f"Target        : {target_col}"
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

        ORDER BY "{date_col}"
        """,
        [
            args.start,
            args.end,
        ],
    ).df()

    if alpha.empty:
        raise RuntimeError(
            "No alpha prediction rows found."
        )

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
        subset=[
            "prediction",
            "target",
        ]
    )

    print(
        f"Alpha rows    : {len(alpha):,}"
    )

    print(
        f"Date start    : "
        f"{alpha['prediction_date'].min()}"
    )

    print(
        f"Date end      : "
        f"{alpha['prediction_date'].max()}"
    )

    # ============================================================
    # 2. Daily Alpha Quality
    #
    # EXACT SAME calculation as the original validation script.
    # ============================================================

    rows = []

    for date, g in alpha.groupby(
        "prediction_date"
    ):

        if len(g) < 5:
            continue

        pred_rank = g["prediction"].rank(
            method="average"
        )

        target_rank = g["target"].rank(
            method="average"
        )

        rank_ic = pred_rank.corr(
            target_rank
        )

        n = len(g)

        k = max(
            1,
            int(n * 0.20),
        )

        ordered = g.sort_values(
            "prediction"
        )

        bottom = (
            ordered
            .head(k)["target"]
            .mean()
        )

        top = (
            ordered
            .tail(k)["target"]
            .mean()
        )

        rows.append(
            {
                "prediction_date": date,
                "rank_ic": rank_ic,
                "top20_return": top,
                "bottom20_return": bottom,
                "top_bottom_spread":
                    top - bottom,
                "alpha_n": n,
            }
        )

    daily = pd.DataFrame(rows)

    if daily.empty:
        raise RuntimeError(
            "Could not calculate daily Alpha statistics."
        )

    daily = (
        daily
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )

    # ============================================================
    # 3. LAGGED Alpha Quality
    #
    # EXACT SAME formula as original.
    #
    # Every rolling statistic is shifted by one prediction date.
    # ============================================================

    daily["rank_ic_mean_5"] = (
        daily["rank_ic"]
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
        .shift(1)
    )

    daily["rank_ic_mean_10"] = (
        daily["rank_ic"]
        .rolling(
            10,
            min_periods=10,
        )
        .mean()
        .shift(1)
    )

    daily["rank_ic_mean_20"] = (
        daily["rank_ic"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
        .shift(1)
    )

    daily["spread_mean_5"] = (
        daily["top_bottom_spread"]
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
        .shift(1)
    )

    daily["spread_mean_10"] = (
        daily["top_bottom_spread"]
        .rolling(
            10,
            min_periods=10,
        )
        .mean()
        .shift(1)
    )

    daily["spread_mean_20"] = (
        daily["top_bottom_spread"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
        .shift(1)
    )

    daily["ic_std_20"] = (
        daily["rank_ic"]
        .rolling(
            20,
            min_periods=20,
        )
        .std()
        .shift(1)
    )

    daily["ic_positive_ratio_10"] = (
        daily["rank_ic"]
        .gt(0)
        .rolling(
            10,
            min_periods=10,
        )
        .mean()
        .shift(1)
    )

    daily["ic_positive_ratio_20"] = (
        daily["rank_ic"]
        .gt(0)
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
        .shift(1)
    )

    # ============================================================
    # 4. Alpha Quality Score
    #
    # EXACT SAME formula as original.
    # ============================================================

    daily["z_ic"] = expanding_z(
        daily["rank_ic_mean_10"]
    )

    daily["z_spread"] = expanding_z(
        daily["spread_mean_10"]
    )

    daily["z_positive_ratio"] = (
        daily["ic_positive_ratio_10"]
        - 0.5
    )

    daily["alpha_quality_score"] = (
        daily["z_ic"].clip(-3, 3)
        + daily["z_spread"].clip(-3, 3)
        + daily["z_positive_ratio"].clip(
            -0.5,
            0.5,
        )
    )

    # ============================================================
    # 5. Save
    #
    # Keep all rows, including initial NaN quality rows.
    # This is important for a full historical series.
    # ============================================================

    daily.to_csv(
        OUT_PATH,
        index=False,
    )

    # ============================================================
    # 6. Verification
    # ============================================================

    print()
    print("=" * 100)
    print("VERIFICATION")
    print("=" * 100)

    print(
        f"Rows                 : {len(daily):,}"
    )

    print(
        f"Date start            : "
        f"{daily['prediction_date'].min()}"
    )

    print(
        f"Date end              : "
        f"{daily['prediction_date'].max()}"
    )

    print()

    for year in [
        2024,
        2025,
        2026,
    ]:

        y = daily[
            daily["prediction_date"].dt.year
            == year
        ]

        print(
            f"{year} rows             : "
            f"{len(y):,}"
        )

        print(
            f"{year} quality non-null : "
            f"{y['alpha_quality_score'].notna().sum():,}"
        )

    # Existing calibration reference.
    q2024 = daily.loc[
        (
            daily["prediction_date"]
            >= pd.Timestamp("2024-01-01")
        )
        &
        (
            daily["prediction_date"]
            <= pd.Timestamp("2024-12-31")
        ),
        "alpha_quality_score",
    ].dropna()

    print()
    print(
        "2024 Q25 from FULL daily series:"
    )

    if not q2024.empty:
        print(
            f"{q2024.quantile(0.25):.6f}"
        )

    print()
    print(
        "2026 quality sample:"
    )

    print(
        daily.loc[
            (
                daily["prediction_date"]
                >= pd.Timestamp("2026-01-01")
            )
            &
            (
                daily["prediction_date"]
                <= pd.Timestamp("2026-09-30")
            ),
            [
                "prediction_date",
                "rank_ic_mean_10",
                "spread_mean_10",
                "ic_positive_ratio_10",
                "alpha_quality_score",
            ],
        ]
        .dropna(
            subset=[
                "alpha_quality_score"
            ]
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"Saved: {OUT_PATH}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
