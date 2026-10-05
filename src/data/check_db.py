import duckdb

con = duckdb.connect("data/market.duckdb")

print("\n=== 테이블 ===")
print(con.execute("SHOW TABLES").df())

print("\n=== 데이터 개수 ===")
print(
    con.execute(
        """
        SELECT
            symbol,
            COUNT(*) AS rows,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM daily_prices
        GROUP BY symbol
        """
    ).df()
)

print("\n=== 최근 10일 ===")
print(
    con.execute(
        """
        SELECT *
        FROM daily_prices
        WHERE symbol = '005930'
        ORDER BY date DESC
        LIMIT 10
        """
    ).df()
)

con.close()
