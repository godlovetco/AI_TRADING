import sys
import time
import duckdb
import pandas as pd

sys.path.append("src/data")

from toss_client import TossClient
from collect_history import candles_to_dataframe


DB_PATH = "data/market.duckdb"
UNIVERSE_FILE = "data/universe_200.csv"

TARGET_ROWS = 1500
PAGE_SIZE = 200

BATCH_SIZE = 50
SLEEP_BETWEEN_SYMBOLS = 0.2


def collect_symbol(client, symbol, name):

    print(f"\n{'=' * 60}")
    print(f"{symbol} {name}")
    print(f"{'=' * 60}")

    all_pages = []
    before = None
    total_rows = 0
    page = 1

    while total_rows < TARGET_ROWS:

        print(f"페이지 {page} 요청 중...")

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

        df = candles_to_dataframe(
            symbol,
            candles
        )

        all_pages.append(df)

        total_rows += len(df)

        print(
            f"  페이지: {len(df)}개 "
            f"/ 누적: {total_rows}개 "
            f"/ {df['date'].min().date()} ~ "
            f"{df['date'].max().date()}"
        )

        next_before = result.get("nextBefore")

        if not next_before:
            print("nextBefore 없음. 수집 종료.")
            break

        before = next_before
        page += 1

    if not all_pages:
        return pd.DataFrame()

    result_df = pd.concat(
        all_pages,
        ignore_index=True
    )

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

    if df.empty:
        return

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

    con.execute(
        "DELETE FROM daily_prices WHERE symbol = ?",
        [symbol]
    )

    con.register(
        "df_view",
        df
    )

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

    universe = pd.read_csv(
        UNIVERSE_FILE,
        dtype={"Code": str}
    )


    universe["Code"] = universe["Code"].str.zfill(6)

    print("=" * 60)
    print("Toss API Universe OHLCV 수집")
    print("=" * 60)

    print(f"Universe: {len(universe)} 종목")
    print(f"목표 데이터: 종목당 최대 {TARGET_ROWS} 거래일")
    print(f"Batch size: {BATCH_SIZE}")

    client = TossClient()
    client.authenticate()

    success = 0
    failed = 0
    total_rows = 0

    start_all = time.time()

    for i, row in universe.iterrows():

        symbol = row["Code"]
        name = row["Name"]

        print()
        print(
            f"[{i + 1}/{len(universe)}] "
            f"{symbol} {name}"
        )

        start = time.time()

        try:

            df = collect_symbol(
                client,
                symbol,
                name
            )

            if df.empty:
                print("데이터 없음")
                failed += 1
                continue

            save_to_db(df)

            elapsed = time.time() - start

            success += 1
            total_rows += len(df)

            print(
                f"저장 완료: {len(df)} rows "
                f"({elapsed:.2f} sec)"
            )

        except Exception as e:

            failed += 1

            print()
            print("!!! ERROR !!!")
            print(e)

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

        # 50종목마다 진행상황
        if (i + 1) % BATCH_SIZE == 0:

            elapsed_all = time.time() - start_all

            print()
            print("=" * 60)
            print(
                f"{i + 1}종목 처리 완료"
            )
            print(
                f"성공: {success}"
            )
            print(
                f"실패: {failed}"
            )
            print(
                f"저장 행: {total_rows:,}"
            )
            print(
                f"경과 시간: {elapsed_all:.1f}초"
            )
            print("=" * 60)

    elapsed_all = time.time() - start_all

    print()
    print("=" * 60)
    print("전체 수집 완료")
    print("=" * 60)

    print(f"성공: {success}")
    print(f"실패: {failed}")
    print(f"총 행 수: {total_rows:,}")
    print(f"총 소요시간: {elapsed_all:.1f}초")


if __name__ == "__main__":
    main()
