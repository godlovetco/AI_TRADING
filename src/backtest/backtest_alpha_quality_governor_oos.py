#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market.duckdb"
QUALITY_PATH = ROOT / "data" / "alpha_quality_lagged_validation.csv"

TRANSACTION_COST = 0.001


def load_v2(con, start, end):
    df = con.execute(
        """
        SELECT *
        FROM alpha_lgbm_risk_v2_beta_backtest
        WHERE prediction_date >= ?
          AND prediction_date <= ?
        ORDER BY prediction_date
        """,
        [start, end],
    ).df()

    for c in ["prediction_date", "entry_date", "exit_date"]:
        df[c] = pd.to_datetime(df[c])

    return df


def calc_metrics(df):
    r = df["net_return"].values

    if len(r) == 0:
        return {}

    equity = np.cumprod(1 + r)

    total_return = equity[-1] - 1

    years = len(r) * 5 / 252

    annual_return = (
        (1 + total_return) ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    annual_vol = np.std(r, ddof=1) * np.sqrt(252 / 5)

    sharpe = (
        np.mean(r) / np.std(r, ddof=1) * np.sqrt(252 / 5)
        if np.std(r, ddof=1) > 0
        else np.nan
    )

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_drawdown = drawdown.min()

    return {
        "periods": len(r),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": np.mean(r > 0),
        "mean_net": np.mean(r),
        "median_net": np.median(r),
        "severe_loss_rate": np.mean(r <= -0.03),
        "worst_trade": np.min(r),
        "best_trade": np.max(r),
    }


def apply_governor(
    df,
    exposure_when_weak,
):
    df = df.sort_values("prediction_date").copy()

    exposures = []
    weak_flags = []

    for _, row in df.iterrows():

        quality = row["alpha_quality_score"]

        if pd.isna(quality):
            weak = False
        else:
            weak = quality <= row["weak_threshold"]

        weak_flags.append(weak)

        if weak:
            exposure = exposure_when_weak
        else:
            exposure = 1.0

        exposures.append(exposure)

    df["weak_alpha"] = weak_flags
    df["exposure"] = exposures

    # ------------------------------------------------------------
    # Approximate portfolio return.
    #
    # Base V2 gross return is scaled by stock exposure.
    # Remaining capital stays in cash.
    #
    # Turnover approximation:
    #   stock turnover scaled by current exposure
    #   + exposure change
    #
    # This is intentionally a first OOS experiment.
    # Exact stock-level turnover requires V2 holdings/weights.
    # ------------------------------------------------------------

    previous_exposure = 0.0

    effective_turnovers = []
    transaction_costs = []
    net_returns = []

    for _, row in df.iterrows():

        exposure = row["exposure"]

        base_gross = row["gross_return"]
        base_turnover = row["turnover"]

        gross_return = exposure * base_gross

        exposure_change = abs(
            exposure - previous_exposure
        )

        effective_turnover = (
            exposure * base_turnover
            + exposure_change
        )

        transaction_cost = (
            effective_turnover * TRANSACTION_COST
        )

        net_return = gross_return - transaction_cost

        effective_turnovers.append(
            effective_turnover
        )

        transaction_costs.append(
            transaction_cost
        )

        net_returns.append(
            net_return
        )

        previous_exposure = exposure

    df["effective_turnover"] = effective_turnovers
    df["governor_transaction_cost"] = transaction_costs
    df["net_return"] = net_returns

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--calibration-start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--calibration-end",
        default="2024-12-31",
    )

    parser.add_argument(
        "--oos-start",
        default="2025-01-01",
    )

    parser.add_argument(
        "--oos-end",
        default="2025-12-31",
    )

    args = parser.parse_args()

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    # ============================================================
    # 1. Load lagged Alpha Quality
    # ============================================================

    quality = pd.read_csv(
        QUALITY_PATH
    )

    quality["prediction_date"] = pd.to_datetime(
        quality["prediction_date"]
    )

    # ============================================================
    # 2. Load V2 trades
    # ============================================================

    calibration = load_v2(
        con,
        args.calibration_start,
        args.calibration_end,
    )

    oos = load_v2(
        con,
        args.oos_start,
        args.oos_end,
    )

    con.close()

    # ============================================================
    # 3. Merge Alpha Quality
    # ============================================================

    qcols = [
        "prediction_date",
        "alpha_quality_score",
        "rank_ic_mean_10",
        "spread_mean_10",
        "ic_positive_ratio_10",
    ]

    quality = quality[qcols].drop_duplicates(
        "prediction_date"
    )

    calibration = calibration.merge(
        quality,
        on="prediction_date",
        how="left",
    )

    oos = oos.merge(
        quality,
        on="prediction_date",
        how="left",
    )

    calibration = calibration.dropna(
        subset=["alpha_quality_score"]
    ).copy()

    oos = oos.dropna(
        subset=["alpha_quality_score"]
    ).copy()

    # ============================================================
    # 4. CALIBRATION
    #
    # Only 2024 is allowed to determine the weak threshold.
    # ============================================================

    weak_threshold = calibration[
        "alpha_quality_score"
    ].quantile(0.25)

    print()
    print("=" * 95)
    print("ALPHA QUALITY GOVERNOR — WALK FORWARD OOS")
    print("=" * 95)

    print(
        f"Calibration period : "
        f"{args.calibration_start} ~ {args.calibration_end}"
    )

    print(
        f"OOS period         : "
        f"{args.oos_start} ~ {args.oos_end}"
    )

    print(
        f"Calibration trades  : {len(calibration)}"
    )

    print(
        f"OOS usable trades   : {len(oos)}"
    )

    print()
    print(
        f"FIXED weak threshold from 2024 Q25 = "
        f"{weak_threshold:.6f}"
    )

    # ============================================================
    # 5. Calibration diagnostic
    # ============================================================

    calibration["weak_alpha"] = (
        calibration["alpha_quality_score"]
        <= weak_threshold
    )

    print()
    print("-" * 95)
    print("2024 CALIBRATION")
    print("-" * 95)

    for flag, g in calibration.groupby(
        "weak_alpha"
    ):

        label = (
            "WEAK"
            if flag
            else "NORMAL"
        )

        print(
            f"{label:8s} "
            f"n={len(g):2d} "
            f"mean_net={g.net_return.mean():+.4%} "
            f"win={np.mean(g.net_return > 0):.2%} "
            f"severe_loss={np.mean(g.net_return <= -0.03):.2%}"
        )

    # ============================================================
    # 6. OOS experiments
    # ============================================================

    results = []
    details = []

    for name, exposure in [
        ("V2_BASELINE", 1.00),
        ("AQG_75", 0.75),
        ("AQG_50", 0.50),
    ]:

        if name == "V2_BASELINE":

            test = oos.copy()

            # Original V2 net return.
            test["net_return"] = test[
                "net_return"
            ]

            test["exposure"] = 1.0
            test["weak_alpha"] = False
            test["effective_turnover"] = test[
                "turnover"
            ]

        else:

            test = oos.copy()

            test["weak_threshold"] = (
                weak_threshold
            )

            test = apply_governor(
                test,
                exposure_when_weak=exposure,
            )

        metrics = calc_metrics(test)

        weak_ratio = test[
            "weak_alpha"
        ].mean()

        mean_exposure = test[
            "exposure"
        ].mean()

        reduced_ratio = np.mean(
            test["exposure"] < 1.0
        )

        metrics.update(
            {
                "strategy": name,
                "weak_trade_ratio": weak_ratio,
                "reduced_ratio": reduced_ratio,
                "mean_exposure": mean_exposure,
                "mean_turnover": test[
                    "effective_turnover"
                ].mean(),
            }
        )

        results.append(metrics)

        test["strategy"] = name

        details.append(test)

    summary = pd.DataFrame(
        results
    )

    detail = pd.concat(
        details,
        ignore_index=True,
    )

    # ============================================================
    # 7. Print OOS results
    # ============================================================

    print()
    print("-" * 95)
    print("2025 OUT-OF-SAMPLE RESULTS")
    print("-" * 95)

    display_cols = [
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
        "reduced_ratio",
        "mean_turnover",
    ]

    print(
        summary[
            display_cols
        ].to_string(index=False)
    )

    # ============================================================
    # 8. Worst trades
    # ============================================================

    print()
    print("-" * 95)
    print("2025 WORST TRADES")
    print("-" * 95)

    worst = detail[
        detail["strategy"] != "V2_BASELINE"
    ].sort_values(
        "net_return"
    ).head(15)

    print(
        worst[
            [
                "strategy",
                "prediction_date",
                "net_return",
                "gross_return",
                "alpha_quality_score",
                "weak_alpha",
                "exposure",
            ]
        ].to_string(index=False)
    )

    # ============================================================
    # 9. Weak Alpha OOS summary
    # ============================================================

    print()
    print("-" * 95)
    print("2025 WEAK ALPHA TRADE ANALYSIS")
    print("-" * 95)

    weak_oos = oos[
        oos["alpha_quality_score"]
        <= weak_threshold
    ]

    normal_oos = oos[
        oos["alpha_quality_score"]
        > weak_threshold
    ]

    for label, g in [
        ("WEAK", weak_oos),
        ("NORMAL", normal_oos),
    ]:

        if len(g) == 0:
            continue

        print(
            f"{label:8s} "
            f"n={len(g):2d} "
            f"mean_net={g.net_return.mean():+.4%} "
            f"median={g.net_return.median():+.4%} "
            f"win={np.mean(g.net_return > 0):.2%} "
            f"severe_loss={np.mean(g.net_return <= -0.03):.2%}"
        )

    # ============================================================
    # 10. Save
    # ============================================================

    summary_path = (
        ROOT
        / "data"
        / "alpha_quality_governor_oos_summary.csv"
    )

    detail_path = (
        ROOT
        / "data"
        / "alpha_quality_governor_oos_detail.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    print()
    print("=" * 95)
    print("SAVED")
    print("=" * 95)

    print(summary_path)
    print(detail_path)

    print()
    print(
        "NOTE: AQG return/cost simulation is a first-stage "
        "OOS approximation because the existing V2 table does "
        "not contain stock-level portfolio weights."
    )


if __name__ == "__main__":
    main()
