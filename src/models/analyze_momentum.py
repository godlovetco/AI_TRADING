import duckdb
import pandas as pd
import numpy as np


DB_PATH = "data/market.duckdb"


def calculate_daily_momentum(df):
    """
    날짜별 RET_20D 기준 Top / Bottom 종목의
    미래 5일 초과수익률을 계산한다.
    """

    results = []

    for date, group in df.groupby("date"):

        group = group.dropna(
            subset=[
                "ret_20d",
                "target_excess_ret_5d"
            ]
        )

        if len(group) < 2:
            continue

        group = group.sort_values(
            "ret_20d",
            ascending=False
        )

        top = group.iloc[0]
        bottom = group.iloc[-1]

        results.append({
            "date": date,
            "top_symbol": top["symbol"],
            "top_ret20": top["ret_20d"],
            "top_target": top["target_excess_ret_5d"],
            "bottom_symbol": bottom["symbol"],
            "bottom_ret20": bottom["ret_20d"],
            "bottom_target": bottom["target_excess_ret_5d"],
        })

    result = pd.DataFrame(results)

    result["spread"] = (
        result["top_target"]
        - result["bottom_target"]
    )

    return result


def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    print_section("RET_20D Momentum Alpha 분석")

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------
    # 1. TEST 데이터 로드
    # --------------------------------------------------

    df = con.execute("""
        SELECT
            symbol,
            date,
            ret_20d,
            target_excess_ret_5d
        FROM features
        WHERE date >= '2025-07-01'
        ORDER BY date, symbol
    """).fetchdf()

    df["date"] = pd.to_datetime(df["date"])

    print(
        f"전체 관측치 : {len(df)}"
    )

    print(
        f"시작일 : {df['date'].min().date()}"
    )

    print(
        f"종료일 : {df['date'].max().date()}"
    )

    # --------------------------------------------------
    # 2. 날짜별 Top / Bottom 계산
    # --------------------------------------------------

    daily = calculate_daily_momentum(df)

    print(
        f"분석 거래일 : {len(daily)}"
    )

    # --------------------------------------------------
    # 3. 전체 성과
    # --------------------------------------------------

    print_section("1. 전체 Momentum 성과")

    print(
        f"Top 평균 초과수익률    : "
        f"{daily['top_target'].mean():.6f}"
    )

    print(
        f"Bottom 평균 초과수익률 : "
        f"{daily['bottom_target'].mean():.6f}"
    )

    print(
        f"Top-Bottom Spread      : "
        f"{daily['spread'].mean():.6f}"
    )

    print(
        f"Top 양수 비율          : "
        f"{(daily['top_target'] > 0).mean():.2%}"
    )

    print(
        f"Bottom 양수 비율       : "
        f"{(daily['bottom_target'] > 0).mean():.2%}"
    )

    print(
        f"Spread 양수 비율       : "
        f"{(daily['spread'] > 0).mean():.2%}"
    )

    # --------------------------------------------------
    # 4. 연도별 성과
    # --------------------------------------------------

    print_section("2. 연도별 Momentum 성과")

    daily["year"] = daily["date"].dt.year

    yearly = (
        daily
        .groupby("year")
        .agg(
            days=("date", "count"),
            top_mean=("top_target", "mean"),
            bottom_mean=("bottom_target", "mean"),
            spread_mean=("spread", "mean"),
            top_positive=(
                "top_target",
                lambda x: (x > 0).mean()
            ),
            bottom_positive=(
                "bottom_target",
                lambda x: (x > 0).mean()
            ),
            spread_positive=(
                "spread",
                lambda x: (x > 0).mean()
            )
        )
    )

    print(
        yearly.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------
    # 5. 월별 성과
    # --------------------------------------------------

    print_section("3. 월별 Momentum 성과")

    daily["month"] = (
        daily["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        daily
        .groupby("month")
        .agg(
            days=("date", "count"),
            top_mean=("top_target", "mean"),
            bottom_mean=("bottom_target", "mean"),
            spread_mean=("spread", "mean"),
            top_positive=(
                "top_target",
                lambda x: (x > 0).mean()
            ),
            spread_positive=(
                "spread",
                lambda x: (x > 0).mean()
            )
        )
    )

    print(
        monthly.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------
    # 6. Top-Bottom Spread 통계
    # --------------------------------------------------

    print_section("4. Top-Bottom Spread 통계")

    spread = daily["spread"]

    print(
        f"평균      : {spread.mean():.6f}"
    )

    print(
        f"중앙값    : {spread.median():.6f}"
    )

    print(
        f"표준편차  : {spread.std():.6f}"
    )

    print(
        f"최소      : {spread.min():.6f}"
    )

    print(
        f"최대      : {spread.max():.6f}"
    )

    # --------------------------------------------------
    # 7. 누적 Spread
    # --------------------------------------------------

    print_section("5. 누적 Top-Bottom Spread")

    daily["cumulative_spread"] = (
        daily["spread"].cumsum()
    )

    print(
        f"최종 누적 Spread : "
        f"{daily['cumulative_spread'].iloc[-1]:.6f}"
    )

    # --------------------------------------------------
    # 8. 최대 Drawdown
    # --------------------------------------------------

    cumulative = daily["cumulative_spread"]

    running_max = cumulative.cummax()

    drawdown = (
        cumulative
        - running_max
    )

    print(
        f"최대 Drawdown : "
        f"{drawdown.min():.6f}"
    )

    # --------------------------------------------------
    # 9. Rolling 60일 Spread
    # --------------------------------------------------

    print_section("6. Rolling 60일 Momentum")

    daily["rolling_60d"] = (
        daily["spread"]
        .rolling(60)
        .mean()
    )

    rolling_valid = daily[
        daily["rolling_60d"].notna()
    ]

    if len(rolling_valid) > 0:

        print(
            f"Rolling 60일 평균 Spread : "
            f"{rolling_valid['rolling_60d'].mean():.6f}"
        )

        print(
            f"Rolling 60일 최소 : "
            f"{rolling_valid['rolling_60d'].min():.6f}"
        )

        print(
            f"Rolling 60일 최대 : "
            f"{rolling_valid['rolling_60d'].max():.6f}"
        )

        print(
            f"Rolling 60일 양수 비율 : "
            f"{(rolling_valid['rolling_60d'] > 0).mean():.2%}"
        )

    # --------------------------------------------------
    # 10. 실제 Top 종목 빈도
    # --------------------------------------------------

    print_section("7. Momentum Top 종목 빈도")

    top_frequency = (
        daily["top_symbol"]
        .value_counts()
        .rename("days")
    )

    top_frequency_pct = (
        top_frequency
        / len(daily)
        * 100
    )

    frequency_df = pd.DataFrame({
        "days": top_frequency,
        "percentage": top_frequency_pct
    })

    print(
        frequency_df.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------
    # 11. Top 종목별 실제 성과
    # --------------------------------------------------

    print_section("8. Momentum Top 종목별 성과")

    symbol_result = (
        daily
        .groupby("top_symbol")
        .agg(
            days=("date", "count"),
            avg_ret20=("top_ret20", "mean"),
            avg_target=("top_target", "mean"),
            positive_ratio=(
                "top_target",
                lambda x: (x > 0).mean()
            )
        )
        .sort_values(
            "avg_target",
            ascending=False
        )
    )

    print(
        symbol_result.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------
    # 12. 최근 20일
    # --------------------------------------------------

    print_section("9. 최근 20거래일 Momentum")

    recent = daily[
        [
            "date",
            "top_symbol",
            "top_ret20",
            "top_target",
            "bottom_symbol",
            "bottom_ret20",
            "bottom_target",
            "spread"
        ]
    ].tail(20)

    print(
        recent.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------
    # 13. 결론용 요약
    # --------------------------------------------------

    print_section("10. Momentum Alpha 요약")

    total_top = daily["top_target"].mean()
    total_bottom = daily["bottom_target"].mean()
    total_spread = daily["spread"].mean()
    spread_positive = (
        daily["spread"] > 0
    ).mean()

    print(
        f"TEST 기간                : "
        f"{daily['date'].min().date()} ~ "
        f"{daily['date'].max().date()}"
    )

    print(
        f"거래일                    : "
        f"{len(daily)}"
    )

    print(
        f"Top 평균 5일 초과수익률   : "
        f"{total_top:.4%}"
    )

    print(
        f"Bottom 평균 5일 초과수익률: "
        f"{total_bottom:.4%}"
    )

    print(
        f"Top-Bottom 평균 Spread    : "
        f"{total_spread:.4%}"
    )

    print(
        f"Spread 양수 비율          : "
        f"{spread_positive:.2%}"
    )

    print()
    print("분석 완료.")

    con.close()


if __name__ == "__main__":
    main()
