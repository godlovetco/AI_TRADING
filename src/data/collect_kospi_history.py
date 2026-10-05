import os
import time
import duckdb
import pandas as pd
import requests


from dotenv import load_dotenv


DB_PATH = "data/market.duckdb"

TARGET_ROWS = 1600
PAGE_SIZE = 200

BASE_URL = "https://openapi.tossinvest.com"


def get_access_token():
    load_dotenv()

    client_id = os.getenv("TOSS_CLIENT_ID")
    client_secret = os.getenv("TOSS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "TOSS_CLIENT_ID 또는 TOSS_CLIENT_SECRET이 없습니다."
        )

    response = requests.post(
        f"{BASE_URL}/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()["access_token"]


def get_kospi_candles(
    access_token,
    count=200,
    before=None,
    max_retries=5,
):
    params = {
        "interval": "1d",
        "count": count,
    }

    if before:
        params["before"] = before

    url = f"{BASE_URL}/api/v1/market-indicators/KOSPI/candles"

    for attempt in range(max_retries):

        response = requests.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            timeout=10,
        )

        # 정상 응답
        if response.status_code == 200:
            return response.json()["result"]

        # 호출 제한
        if response.status_code == 429:

            wait_seconds = 5 * (attempt + 1)

            print(
                f"  API 호출 제한(429) 발생 "
                f"→ {wait_seconds}초 후 재시도..."
            )

            time.sleep(wait_seconds)
            continue

        # 그 외 오류
        print("API ERROR:")
        print(response.text)

        response.raise_for_status()

    raise RuntimeError(
        "API 호출 제한으로 최대 재시도 횟수를 초과했습니다."
    )

def candles_to_dataframe(candles):
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

    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    return df


def collect_history(access_token):
    all_pages = []

    before = None
    total_rows = 0
    page = 1

    print("=== KOSPI 장기 데이터 수집 ===")
    print(f"목표 데이터: 약 {TARGET_ROWS} 거래일")

    while total_rows < TARGET_ROWS:

        print(f"\n페이지 {page} 요청 중...")

        result = get_kospi_candles(
            access_token=access_token,
            count=PAGE_SIZE,
            before=before,
        )

        candles = result["candles"]

        if not candles:
            print("더 이상 데이터가 없습니다.")
            break

        df = candles_to_dataframe(candles)

        all_pages.append(df)

        total_rows += len(df)

        print(f"  이번 페이지 : {len(df)}개")
        print(f"  누적 데이터 : {total_rows}개")
        print(
            f"  날짜 범위   : "
            f"{df['date'].min()} ~ {df['date'].max()}"
        )

        next_before = result.get("nextBefore")

        if not next_before:
            print("nextBefore가 없어 수집 종료.")
            break

        before = next_before
        page += 1

        # Toss API 호출 제한 방지
        time.sleep(2)


    if not all_pages:
        return pd.DataFrame()

    df = pd.concat(
        all_pages,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def save_to_db(df):
    con = duckdb.connect(DB_PATH)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kospi_daily (
            date TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT
        )
        """
    )

    # 기존 KOSPI 데이터 제거
    con.execute("DELETE FROM kospi_daily")

    con.register("df_view", df)

    con.execute(
        """
        INSERT INTO kospi_daily
        SELECT
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

    access_token = get_access_token()

    df = collect_history(access_token)

    if df.empty:
        print("KOSPI 데이터를 가져오지 못했습니다.")
        return

    save_to_db(df)

    print("\n" + "=" * 50)
    print("KOSPI 저장 완료")
    print("=" * 50)

    print(f"행 수  : {len(df)}")
    print(f"시작일 : {df['date'].min()}")
    print(f"종료일 : {df['date'].max()}")


if __name__ == "__main__":
    main()
