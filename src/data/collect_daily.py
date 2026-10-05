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

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    df["symbol"] = symbol

    df = df[
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

    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_daily_prices(client, symbol, count=200):
    print(f"\n[{symbol}] {SYMBOLS[symbol]} 데이터 수집 중...")

    result = client.get_daily_candles(
        symbol,
        count=count,
        adjusted=True,
    )

    candles = result["candles"]

    df = candles_to_dataframe(symbol, candles)

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

    # 해당 종목의 기존 데이터를 삭제하고 최신 데이터로 교체
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

    print(f"  저장 건수 : {len(df)}")
    print(f"  시작일    : {df['date'].min()}")
    print(f"  종료일    : {df['date'].max()}")

    return len(df)


def main():
    client = TossClient()

    print("=== Toss증권 5종목 일봉 수집 ===")

    client.authenticate()

    total = 0

    for symbol in SYMBOLS:
        try:
            rows = save_daily_prices(
                client,
                symbol,
                count=200,
            )
            total += rows

        except Exception as e:
            print(f"  오류 발생: {symbol} / {e}")

    print("\n=== 수집 완료 ===")
    print(f"총 저장 행 수: {total}")


if __name__ == "__main__":
    main()
