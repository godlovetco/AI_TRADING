import duckdb
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def evaluate_model(model, X, y, name):

    prediction = model.predict(X)

    mae = mean_absolute_error(y, prediction)

    rmse = np.sqrt(
        mean_squared_error(y, prediction)
    )

    r2 = r2_score(y, prediction)

    correlation = np.corrcoef(
        y,
        prediction
    )[0, 1]

    print(f"\n[{name}]")

    print(f"MAE         : {mae:.6f}")
    print(f"RMSE        : {rmse:.6f}")
    print(f"R²          : {r2:.6f}")
    print(f"Correlation : {correlation:.6f}")

    return prediction


def main():

    print("=" * 60)
    print("Mini Wall Street AI - Alpha Model V1")
    print("=" * 60)

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------
    # 데이터 불러오기
    # --------------------------------------------------

    train = con.execute("""
        SELECT *
        FROM ml_train
        ORDER BY date, symbol
    """).fetchdf()

    validation = con.execute("""
        SELECT *
        FROM ml_validation
        ORDER BY date, symbol
    """).fetchdf()

    test = con.execute("""
        SELECT *
        FROM ml_test
        ORDER BY date, symbol
    """).fetchdf()

    print("\n데이터 크기")

    print(f"TRAIN      : {len(train)}")
    print(f"VALIDATION : {len(validation)}")
    print(f"TEST       : {len(test)}")

    # --------------------------------------------------
    # X / y
    # --------------------------------------------------

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # --------------------------------------------------
    # 모델
    # --------------------------------------------------

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )

    print("\n모델 학습 시작...")

    model.fit(
        X_train,
        y_train
    )

    print("모델 학습 완료")

    # --------------------------------------------------
    # 평가
    # --------------------------------------------------

    train_pred = evaluate_model(
        model,
        X_train,
        y_train,
        "TRAIN"
    )

    validation_pred = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION"
    )

    test_pred = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST"
    )

    # --------------------------------------------------
    # 예측값 저장
    # --------------------------------------------------

    test_result = test[
        [
            "symbol",
            "date",
            "close",
            TARGET_COLUMN
        ]
    ].copy()

    test_result["prediction"] = test_pred

    test_result["prediction_rank"] = (
        test_result
        .groupby("date")["prediction"]
        .rank(
            ascending=False,
            method="first"
        )
    )

    print("\n" + "=" * 60)
    print("TEST 예측 결과 샘플")
    print("=" * 60)

    latest_date = test_result["date"].max()

    latest = test_result[
        test_result["date"] == latest_date
    ].sort_values(
        "prediction",
        ascending=False
    )

    print(
        latest.to_string(index=False)
    )

    # --------------------------------------------------
    # DB 저장
    # --------------------------------------------------

    con.execute("""
        DROP TABLE IF EXISTS alpha_predictions
    """)

    con.register(
        "test_result_df",
        test_result
    )

    con.execute("""
        CREATE TABLE alpha_predictions AS
        SELECT *
        FROM test_result_df
    """)

    print("\nDB 저장 완료:")
    print("alpha_predictions")

    con.close()


if __name__ == "__main__":
    main()
