import FinanceDataReader as fdr
import pandas as pd


UNIVERSE_SIZE = 200


def main():

    print("=" * 60)
    print("KRX AI Trading Universe 생성")
    print("=" * 60)

    df = fdr.StockListing("KRX")

    print(f"전체 KRX 종목: {len(df)}")

    # 일반 주식
    u = df[
        (df["MarketId"] == "STK")
        & (df["Market"].isin(["KOSPI", "KOSDAQ"]))
    ].copy()

    print(f"KOSPI/KOSDAQ 일반주식: {len(u)}")

    # 우선주 제외
    u = u[
        ~u["Name"].str.endswith("우", na=False)
    ].copy()

    print(f"우선주 제외 후: {len(u)}")

    # 거래대금 기준 정렬
    u = u.sort_values(
        "Amount",
        ascending=False
    ).reset_index(drop=True)

    # 상위 200
    universe = u.head(UNIVERSE_SIZE).copy()

    # Universe Rank
    universe["universe_rank"] = range(
        1,
        len(universe) + 1
    )

    # 필요한 컬럼만
    universe = universe[
        [
            "universe_rank",
            "Code",
            "ISU_CD",
            "Name",
            "Market",
            "Close",
            "Amount",
            "Marcap",
            "Stocks",
        ]
    ]

    # 저장
    output = "data/universe_200.csv"

    universe.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("Universe 생성 완료")
    print("=" * 60)

    print(f"종목 수: {len(universe)}")
    print(f"파일: {output}")

    print()
    print("=== 상위 20 ===")
    print(
        universe.head(20).to_string(
            index=False
        )
    )

    print()
    print("=== 마지막 10개 ===")
    print(
        universe.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
