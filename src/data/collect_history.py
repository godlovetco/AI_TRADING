import duckdb
import pandas as pd

from toss_client import TossClient


DB_PATH = "data/market.duckdb"

SYMBOLS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "NAVER",
    "035720": "카카오",
}

TARGET_ROWS = 1500
PAGE_SIZE = 200


def candles_to_dataframe(symbol, candles):
    df = pd.DataFrame(candles)

    df = df.rename(
        columns={
            "timestamp": "date",
            "openPrice": "open",
            "highPrice": "high",
            "lowPrice": "low",
            "closePrice": "close",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(df[column])

    df["symbol"] = symbol

    return df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def collect_symbol(client, symbol):
    name = SYMBOLS[symbol]

    print(f"\n{'=' * 50}")
    print(f"{symbol} {name}")
    print(f"{'=' * 50}")

    all_pages = []
    before = None
    total_rows = 0
    page = 1

    while total_rows < TARGET_ROWS:
        print(f"\n페이지 {page} 요청 중...")

        result = client.get_daily_candles(
            symbol=symbol,
            count=PAGE_SIZE,
            before=before,
            adjusted=True,
        )

        candles = result["candles"]

        if not candles:
            print("더 이상 데이터가 없습니다.")
            break

        df = candles_to_dataframe(symbol, candles)

        all_pages.append(df)

        total_rows += len(df)

        print(f"  이번 페이지 : {len(df)}개")
        print(f"  누적 데이터 : {total_rows}개")
        print(f"  날짜 범위   : {df['date'].min()} ~ {df['date'].max()}")

        next_before = result.get("nextBefore")

        if not next_before:
            print("nextBefore가 없어 수집 종료.")
            break

        before = next_before
        page += 1

    if not all_pages:
        return pd.DataFrame()

    result_df = pd.concat(
        all_pages,
        ignore_index=True,
    )

    # 중복 제거
    result_df = (
        result_df
        .drop_duplicates(
            subset=["symbol", "date"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return result_df


def save_to_db(df):
    con = duckdb.connect(DB_PATH)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol VARCHAR,
            date TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT
        )
        """
    )

    symbol = df["symbol"].iloc[0]

    # 해당 종목 기존 데이터 삭제
    con.execute(
        "DELETE FROM daily_prices WHERE symbol = ?",
        [symbol],
    )

    con.register("df_view", df)

    con.execute(
        """
        INSERT INTO daily_prices
        SELECT
            symbol,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM df_view
        """
    )

    con.close()


def main():
    client = TossClient()

    print("=== Toss증권 장기 일봉 데이터 수집 ===")
    print(f"목표 데이터: 종목당 약 {TARGET_ROWS} 거래일")

    client.authenticate()

    total = 0

    for symbol in SYMBOLS:
        try:
            df = collect_symbol(
                client,
                symbol,
            )

            if df.empty:
                print(f"{symbol}: 데이터 없음")
                continue

            save_to_db(df)

            print("\n저장 완료")
            print(f"  종목     : {symbol}")
            print(f"  행 수    : {len(df)}")
            print(f"  시작일   : {df['date'].min()}")
            print(f"  종료일   : {df['date'].max()}")

            total += len(df)

        except Exception as e:
            print(f"\n!!! {symbol} 오류 !!!")
            print(e)

    print("\n" + "=" * 50)
    print("전체 수집 완료")
    print(f"총 저장 행 수: {total}")
    print("=" * 50)


if __name__ == "__main__":
    main()
