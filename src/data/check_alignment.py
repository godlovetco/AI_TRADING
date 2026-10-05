import duckdb
import pandas as pd

DB_PATH = "data/market.duckdb"

con = duckdb.connect(DB_PATH)

print("=" * 60)
print("DB 날짜 정합성 검사")
print("=" * 60)

# 1. KOSPI 기본 정보
kospi = con.execute("""
    SELECT
        COUNT(*) AS rows,
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(DISTINCT date) AS unique_dates
    FROM kospi_daily
""").fetchdf()

print("\n[KOSPI]")
print(kospi.to_string(index=False))

# 2. 종목별 날짜 정보
stocks = con.execute("""
    SELECT
        symbol,
        COUNT(*) AS rows,
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(DISTINCT date) AS unique_dates
    FROM daily_prices
    GROUP BY symbol
    ORDER BY symbol
""").fetchdf()

print("\n[종목별]")
print(stocks.to_string(index=False))

# 3. 전체 종목과 KOSPI의 공통 거래일 수
alignment = con.execute("""
    SELECT
        d.symbol,
        COUNT(*) AS common_dates
    FROM daily_prices d
    INNER JOIN kospi_daily k
        ON d.date = k.date
    GROUP BY d.symbol
    ORDER BY d.symbol
""").fetchdf()

print("\n[종목 ↔ KOSPI 공통 거래일]")
print(alignment.to_string(index=False))

# 4. 날짜가 서로 다른 경우 확인
print("\n[종목별 날짜 불일치 검사]")

for symbol in stocks["symbol"]:
    missing_stock = con.execute("""
        SELECT COUNT(*)
        FROM kospi_daily k
        LEFT JOIN daily_prices d
            ON k.date = d.date
           AND d.symbol = ?
        WHERE d.date IS NULL
    """, [symbol]).fetchone()[0]

    missing_kospi = con.execute("""
        SELECT COUNT(*)
        FROM daily_prices d
        LEFT JOIN kospi_daily k
            ON d.date = k.date
        WHERE d.symbol = ?
          AND k.date IS NULL
    """, [symbol]).fetchone()[0]

    print(
        f"{symbol}: "
        f"KOSPI에 없는 종목 날짜 = {missing_stock}, "
        f"종목에 없는 KOSPI 날짜 = {missing_kospi}"
    )

print("\n" + "=" * 60)
print("검사 완료")
print("=" * 60)

con.close()
