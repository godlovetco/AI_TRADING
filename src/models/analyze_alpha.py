import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/market.duckdb"

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "volatility_20d",
    "volume_ratio_20d",
    "rsi_14",
    "atr_14",
    "kospi_ret_1d",
    "kospi_ret_5d",
    "kospi_ret_20d",
    "kospi_volatility_20d",
]

TARGET = "target_excess_ret_5d"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    print_section("Alpha Model 진단")

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------
    # 1. Feature + Target
    # --------------------------------------------------

    df = con.execute("""
        SELECT
            symbol,
            date,
            target_excess_ret_5d,
            ret_1d,
            ret_5d,
            ret_20d,
            ma5_ratio,
            ma20_ratio,
            ma60_ratio,
            volatility_20d,
            volume_ratio_20d,
            rsi_14,
            atr_14,
            kospi_ret_1d,
            kospi_ret_5d,
            kospi_ret_20d,
            kospi_volatility_20d
        FROM features
        ORDER BY date, symbol
    """).fetchdf()

    # --------------------------------------------------
    # 2. Target 기본 통계
    # --------------------------------------------------

    print_section("1. Target 기본 통계")

    target_stats = df[TARGET].describe()

    print(target_stats.to_string())

    print(
        f"\nTarget 양수 비율 : "
        f"{(df[TARGET] > 0).mean():.2%}"
    )

    print(
        f"Target 음수 비율 : "
        f"{(df[TARGET] < 0).mean():.2%}"
    )

    print(
        f"Target 0 비율    : "
        f"{(df[TARGET] == 0).mean():.2%}"
    )

    # --------------------------------------------------
    # 3. 종목별 Target 통계
    # --------------------------------------------------

    print_section("2. 종목별 Target 통계")

    stock_stats = (
        df.groupby("symbol")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            positive_ratio=lambda x: (x > 0).mean()
        )
        .sort_values("mean", ascending=False)
    )

    print(stock_stats.to_string())

    # --------------------------------------------------
    # 4. 연도별 Target
    # --------------------------------------------------

    print_section("3. 연도별 Target 통계")

    df["year"] = pd.to_datetime(df["date"]).dt.year

    yearly = (
        df.groupby("year")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            positive_ratio=lambda x: (x > 0).mean()
        )
    )

    print(yearly.to_string())

    # --------------------------------------------------
    # 5. Feature ↔ Target 상관관계
    # --------------------------------------------------

    print_section("4. Feature ↔ Target 상관관계")

    correlations = []

    for feature in FEATURE_COLUMNS:

        corr = df[feature].corr(
            df[TARGET]
        )

        correlations.append(
            {
                "feature": feature,
                "correlation": corr
            }
        )

    correlation_df = (
        pd.DataFrame(correlations)
        .sort_values(
            "correlation",
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(
        correlation_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 6. Alpha Prediction 분석
    # --------------------------------------------------

    print_section("5. Model Prediction 분석")

    predictions = con.execute("""
        SELECT
            symbol,
            date,
            target_excess_ret_5d,
            prediction,
            prediction_rank
        FROM alpha_predictions
        ORDER BY date, symbol
    """).fetchdf()

    print(
        f"Prediction 데이터 : "
        f"{len(predictions)}행"
    )

    prediction_corr = predictions[
        "prediction"
    ].corr(
        predictions[TARGET]
    )

    print(
        f"\nPrediction ↔ 실제 Target 상관관계 : "
        f"{prediction_corr:.6f}"
    )

    # --------------------------------------------------
    # 7. Prediction 분위수별 실제 성과
    # --------------------------------------------------

    print_section("6. Prediction 분위수별 실제 성과")

    predictions["prediction_quantile"] = (
        predictions.groupby("date")["prediction"]
        .transform(
            lambda x: pd.qcut(
                x.rank(method="first"),
                q=5,
                labels=False
            ) + 1
        )
    )

    quantile_result = (
        predictions
        .groupby("prediction_quantile")[
            TARGET
        ]
        .agg(
            count="count",
            mean="mean",
            median="median",
            positive_ratio=lambda x: (x > 0).mean()
        )
    )

    print(quantile_result.to_string())

    # --------------------------------------------------
    # 8. 날짜별 Top 1 성과
    # --------------------------------------------------

    print_section("7. 매일 Prediction 1위 종목의 실제 성과")

    top1 = (
        predictions
        .sort_values(
            ["date", "prediction"],
            ascending=[True, False]
        )
        .groupby("date")
        .head(1)
    )

    print(
        f"Top 1 관측 수 : {len(top1)}"
    )

    print(
        f"Top 1 평균 5일 초과수익률 : "
        f"{top1[TARGET].mean():.6f}"
    )

    print(
        f"Top 1 양수 비율 : "
        f"{(top1[TARGET] > 0).mean():.2%}"
    )

    # --------------------------------------------------
    # 9. 날짜별 Bottom 1 성과
    # --------------------------------------------------

    bottom1 = (
        predictions
        .sort_values(
            ["date", "prediction"],
            ascending=[True, True]
        )
        .groupby("date")
        .head(1)
    )

    print(
        f"\nBottom 1 평균 5일 초과수익률 : "
        f"{bottom1[TARGET].mean():.6f}"
    )

    print(
        f"Bottom 1 양수 비율 : "
        f"{(bottom1[TARGET] > 0).mean():.2%}"
    )

    # --------------------------------------------------
    # 10. 최신 TEST 예측
    # --------------------------------------------------

    print_section("8. 최신 TEST Prediction")

    latest_date = predictions["date"].max()

    latest = (
        predictions[
            predictions["date"] == latest_date
        ]
        .sort_values(
            "prediction",
            ascending=False
        )
    )

    print(
        latest.to_string(
            index=False
        )
    )

    con.close()

    print_section("진단 완료")


if __name__ == "__main__":
    main()
