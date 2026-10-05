from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma120_ratio",
    "volatility_20d",
    "volatility_60d",
    "volume_ratio_20d",
    "rsi_14",
    "atr_14",
    "atr_ratio_14",
    "kospi_return_1d",
    "kospi_return_5d",
    "kospi_return_20d",
    "kospi_return_60d",
    "relative_strength_5d",
    "relative_strength_20d",
    "relative_strength_60d",
]

TARGET_COLUMN = "future_excess_return_5d"


def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    columns = ["symbol", "date"] + FEATURE_COLUMNS + [TARGET_COLUMN]

    query = f"""
        SELECT {", ".join(columns)}
        FROM stock_features
        WHERE {TARGET_COLUMN} IS NOT NULL
        ORDER BY date, symbol
    """

    df = con.execute(query).fetchdf()
    con.close()

    df["date"] = pd.to_datetime(df["date"])

    return df


def split_data(df):
    train = df[
        (df["date"] >= "2020-09-14") &
        (df["date"] <= "2024-12-31")
    ].copy()

    validation = df[
        (df["date"] >= "2025-01-01") &
        (df["date"] <= "2025-12-31")
    ].copy()

    test = df[
        (df["date"] >= "2026-01-01") &
        (df["date"] <= "2026-09-30")
    ].copy()

    return train, validation, test


def calculate_rank_ic(pred, actual):
    x = pd.Series(pred).rank(method="average")
    y = pd.Series(actual).rank(method="average")

    return x.corr(y)


def daily_rank_ic(df, pred_col="prediction"):
    values = []

    for date, group in df.groupby("date"):
        if len(group) < 10:
            continue

        ic = calculate_rank_ic(
            group[pred_col].values,
            group[TARGET_COLUMN].values,
        )

        if pd.notna(ic):
            values.append(ic)

    return pd.Series(values)


def evaluate(name, df):
    y_true = df[TARGET_COLUMN].to_numpy()
    y_pred = df["prediction"].to_numpy()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    ic_series = daily_rank_ic(df)

    mean_ic = ic_series.mean()
    ic_std = ic_series.std(ddof=1)

    if ic_std > 0:
        ic_ir = mean_ic / ic_std * np.sqrt(252)
    else:
        ic_ir = np.nan

    positive_ic = (ic_series > 0).mean()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows              : {len(df):,}")
    print(f"RMSE              : {rmse:.6f}")
    print(f"R2                : {r2:.6f}")
    print(f"Pearson Corr      : {corr:.6f}")
    print(f"Mean Rank IC      : {mean_ic:.6f}")
    print(f"IC Std            : {ic_std:.6f}")
    print(f"IC IR             : {ic_ir:.6f}")
    print(f"Positive IC       : {positive_ic:.2%}")

    return {
        "period": name,
        "rows": len(df),
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "positive_ic": positive_ic,
    }


def top_bottom_spread(df, quantile=0.2):
    spreads = []

    for date, group in df.groupby("date"):
        if len(group) < 20:
            continue

        group = group.sort_values("prediction")

        n = max(int(len(group) * quantile), 1)

        bottom = group.iloc[:n][TARGET_COLUMN].mean()
        top = group.iloc[-n:][TARGET_COLUMN].mean()

        spreads.append(top - bottom)

    spreads = pd.Series(spreads)

    return {
        "mean_spread": spreads.mean(),
        "std_spread": spreads.std(ddof=1),
        "positive": (spreads > 0).mean(),
        "days": len(spreads),
    }


def main():
    print("=" * 70)
    print("LightGBM Alpha Model")
    print("=" * 70)

    df = load_data()

    print(f"Total rows    : {len(df):,}")
    print(f"Symbols       : {df['symbol'].nunique()}")
    print(f"Date range    : {df['date'].min().date()} ~ {df['date'].max().date()}")

    train, validation, test = split_data(df)

    print()
    print("Data split")
    print("-" * 70)
    print(
        f"Train      : {train['date'].min().date()} ~ "
        f"{train['date'].max().date()} "
        f"({len(train):,})"
    )
    print(
        f"Validation : {validation['date'].min().date()} ~ "
        f"{validation['date'].max().date()} "
        f"({len(validation):,})"
    )
    print(
        f"Test       : {test['date'].min().date()} ~ "
        f"{test['date'].max().date()} "
        f"({len(test):,})"
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print()
    print("Training LightGBM...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(100, verbose=True),
            log_evaluation(100),
        ],
    )

    print()
    print(f"Best iteration: {model.best_iteration_}")

    train["prediction"] = model.predict(X_train)
    validation["prediction"] = model.predict(X_validation)
    test["prediction"] = model.predict(X_test)

    results = []

    results.append(evaluate("TRAIN", train))
    results.append(evaluate("VALIDATION", validation))
    results.append(evaluate("TEST", test))

    print()
    print("=" * 70)
    print("Top 20% vs Bottom 20%")
    print("=" * 70)

    for name, data in [
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        result = top_bottom_spread(data)

        print()
        print(name)
        print(f"Trading days       : {result['days']}")
        print(f"Mean Top-Bottom    : {result['mean_spread']:.6f}")
        print(f"Spread Std         : {result['std_spread']:.6f}")
        print(f"Positive Spread    : {result['positive']:.2%}")

    print()
    print("=" * 70)
    print("Feature Importance")
    print("=" * 70)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(importance.to_string(index=False))

    # Save predictions
    output = pd.concat([
        train[["symbol", "date", TARGET_COLUMN, "prediction"]],
        validation[["symbol", "date", TARGET_COLUMN, "prediction"]],
        test[["symbol", "date", TARGET_COLUMN, "prediction"]],
    ])

    con = duckdb.connect(str(DB_PATH))

    con.execute("DROP TABLE IF EXISTS alpha_lgbm_predictions")

    con.register("output_df", output)

    con.execute("""
        CREATE TABLE alpha_lgbm_predictions AS
        SELECT *
        FROM output_df
    """)

    con.register("importance_df", importance)

    con.execute("DROP TABLE IF EXISTS alpha_lgbm_importance")

    con.execute("""
        CREATE TABLE alpha_lgbm_importance AS
        SELECT *
        FROM importance_df
    """)

    con.close()

    print()
    print("=" * 70)
    print("Saved")
    print("=" * 70)
    print("alpha_lgbm_predictions")
    print("alpha_lgbm_importance")


if __name__ == "__main__":
    main()