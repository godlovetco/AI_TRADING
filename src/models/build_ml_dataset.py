import duckdb
import pandas as pd

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

TARGET_COLUMN = "target_excess_ret_5d"


def main():

    print("=" * 60)
    print("ML Dataset 생성")
    print("=" * 60)

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------
    # Feature 데이터 불러오기
    # --------------------------------------------------

    df = con.execute("""
        SELECT
            symbol,
            date,
            close,
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
            kospi_volatility_20d,
            target_excess_ret_5d
        FROM features
        ORDER BY date, symbol
    """).fetchdf()

    print(f"\n전체 데이터 : {len(df)}행")

    # --------------------------------------------------
    # 날짜 확인
    # --------------------------------------------------

    print("\n전체 기간:")
    print(f"시작일 : {df['date'].min()}")
    print(f"종료일 : {df['date'].max()}")

    # --------------------------------------------------
    # Feature / Target 결측치 검사
    # --------------------------------------------------

    print("\n결측치 검사:")

    check_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = df[check_columns].isnull().sum()

    print(missing[missing > 0])

    if missing.sum() == 0:
        print("결측치 없음")

    # --------------------------------------------------
    # 날짜 기준 Train / Validation / Test 분리
    # --------------------------------------------------

    train_end = pd.Timestamp("2024-12-31")
    validation_end = pd.Timestamp("2025-06-30")

    train = df[df["date"] <= train_end].copy()

    validation = df[
        (df["date"] > train_end)
        & (df["date"] <= validation_end)
    ].copy()

    test = df[
        df["date"] > validation_end
    ].copy()

    print("\n" + "=" * 60)
    print("시간 순서 데이터 분리")
    print("=" * 60)

    print("\nTRAIN")
    print(f"기간 : {train['date'].min()} ~ {train['date'].max()}")
    print(f"행 수 : {len(train)}")

    print("\nVALIDATION")
    print(
        f"기간 : "
        f"{validation['date'].min()} ~ "
        f"{validation['date'].max()}"
    )
    print(f"행 수 : {len(validation)}")

    print("\nTEST")
    print(f"기간 : {test['date'].min()} ~ {test['date'].max()}")
    print(f"행 수 : {len(test)}")

    # --------------------------------------------------
    # 종목별 데이터 확인
    # --------------------------------------------------

    print("\nTRAIN 종목별:")
    print(train.groupby("symbol").size().to_string())

    print("\nVALIDATION 종목별:")
    print(validation.groupby("symbol").size().to_string())

    print("\nTEST 종목별:")
    print(test.groupby("symbol").size().to_string())

    # --------------------------------------------------
    # 날짜 겹침 검사
    # --------------------------------------------------

    train_dates = set(train["date"])
    validation_dates = set(validation["date"])
    test_dates = set(test["date"])

    print("\n" + "=" * 60)
    print("데이터 누수 검사")
    print("=" * 60)

    print(
        "TRAIN ↔ VALIDATION 날짜 중복:",
        len(train_dates & validation_dates)
    )

    print(
        "VALIDATION ↔ TEST 날짜 중복:",
        len(validation_dates & test_dates)
    )

    print(
        "TRAIN ↔ TEST 날짜 중복:",
        len(train_dates & test_dates)
    )

    # --------------------------------------------------
    # ML 데이터 저장
    # --------------------------------------------------

    con.execute("""
        DROP TABLE IF EXISTS ml_train
    """)

    con.execute("""
        DROP TABLE IF EXISTS ml_validation
    """)

    con.execute("""
        DROP TABLE IF EXISTS ml_test
    """)

    con.register("train_df", train)
    con.register("validation_df", validation)
    con.register("test_df", test)

    con.execute("""
        CREATE TABLE ml_train AS
        SELECT * FROM train_df
    """)

    con.execute("""
        CREATE TABLE ml_validation AS
        SELECT * FROM validation_df
    """)

    con.execute("""
        CREATE TABLE ml_test AS
        SELECT * FROM test_df
    """)

    print("\n" + "=" * 60)
    print("ML Dataset 저장 완료")
    print("=" * 60)

    print("ml_train")
    print("ml_validation")
    print("ml_test")

    con.close()


if __name__ == "__main__":
    main()
