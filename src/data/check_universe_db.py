import duckdb

DB_PATH = "data/market.duckdb"


def main():

    con = duckdb.connect(DB_PATH)

    print("=" * 70)
    print("Universe DB 무결성 검사")
    print("=" * 70)

    # 1. 전체 종목/행 수
    result = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT symbol) AS symbols,
            COUNT(DISTINCT date) AS trading_days
        FROM daily_prices
        """
    ).fetchone()

    print()
    print("=== 전체 DB ===")
    print(f"전체 행 수     : {result[0]:,}")
    print(f"종목 수        : {result[1]:,}")
    print(f"거래일 수      : {result[2]:,}")

    # 2. 종목별 데이터
    print()
    print("=== 종목별 데이터 ===")

    rows = con.execute(
        """
        SELECT
            symbol,
            COUNT(*) AS rows,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(DISTINCT date) AS unique_dates
        FROM daily_prices
        GROUP BY symbol
        ORDER BY symbol
        """
    ).fetchall()

    for row in rows:
        print(
            f"{row[0]}  "
            f"{row[1]:4d} rows  "
            f"{row[2].date()} ~ {row[3].date()}  "
            f"unique={row[4]}"
        )

    # 3. 중복 검사
    duplicate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                symbol,
                date,
                COUNT(*) AS cnt
            FROM daily_prices
            GROUP BY symbol, date
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    print()
    print("=== 중복 검사 ===")
    print(f"중복 symbol-date : {duplicate_count}")

    # 4. NULL 검사
    null_count = con.execute(
        """
        SELECT COUNT(*)
        FROM daily_prices
        WHERE
            symbol IS NULL
            OR date IS NULL
            OR open IS NULL
            OR high IS NULL
            OR low IS NULL
            OR close IS NULL
            OR volume IS NULL
        """
    ).fetchone()[0]

    print()
    print("=== NULL 검사 ===")
    print(f"NULL 행 : {null_count}")

    # 5. 가격 이상치
    bad_price_count = con.execute(
        """
        SELECT COUNT(*)
        FROM daily_prices
        WHERE
            open <= 0
            OR high <= 0
            OR low <= 0
            OR close <= 0
            OR high < low
        """
    ).fetchone()[0]

    print()
    print("=== 가격 이상 검사 ===")
    print(f"이상 가격 행 : {bad_price_count}")

    con.close()

    print()
    print("=" * 70)
    print("검사 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
