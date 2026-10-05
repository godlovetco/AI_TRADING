import duckdb
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


DB_PATH = "data/market.duckdb"

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

TARGET_RAW = "future_excess_return_5d"

TRAIN_END = "2024-12-31"
VALIDATION_END = "2025-12-31"


def make_cross_sectional_rank(df):
    """
    각 거래일별로 future_excess_return_5d의
    cross-sectional percentile rank를 계산한다.

    0.0 = 해당 날짜의 최하위
    1.0 = 해당 날짜의 최상위
    """

    df = df.copy()

    df["target_rank_5d"] = (
        df.groupby("date")[TARGET_RAW]
        .rank(method="average", pct=True)
    )

    return df


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = pearsonr(y_true, y_pred)[0]
    else:
        corr = np.nan

    return rmse, r2, corr


def daily_rank_ic(df):
    """
    날짜별 Spearman Rank IC.
    """

    records = []

    for date, g in df.groupby("date"):
        if len(g) < 10:
            continue

        if g["prediction"].nunique() < 2:
            continue

        if g["target_rank_5d"].nunique() < 2:
            continue

        ic = spearmanr(
            g["prediction"],
            g["target_rank_5d"]
        ).statistic

        if np.isfinite(ic):
            records.append((date, ic))

    ic_df = pd.DataFrame(records, columns=["date", "ic"])

    if ic_df.empty:
        return ic_df

    return ic_df


def top_bottom_spread(df, top_pct=0.20):
    """
    각 거래일 기준 prediction 상위/하위 종목의
    실제 future_excess_return_5d 차이를 계산한다.
    """

    records = []

    for date, g in df.groupby("date"):

        if len(g) < 10:
            continue

        g = g.sort_values("prediction")

        n = max(1, int(len(g) * top_pct))

        bottom = g.iloc[:n]
        top = g.iloc[-n:]

        top_return = top[TARGET_RAW].mean()
        bottom_return = bottom[TARGET_RAW].mean()

        spread = top_return - bottom_return

        records.append(
            (
                date,
                top_return,
                bottom_return,
                spread,
            )
        )

    return pd.DataFrame(
        records,
        columns=[
            "date",
            "top_return",
            "bottom_return",
            "spread",
        ],
    )


def print_metrics(name, df):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    y_true = df["target_rank_5d"].values
    y_pred = df["prediction"].values

    rmse, r2, corr = evaluate_regression(
        y_true,
        y_pred
    )

    ic_df = daily_rank_ic(df)

    mean_ic = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()

    if ic_std > 0:
        ic_ir = mean_ic / ic_std * np.sqrt(252)
    else:
        ic_ir = np.nan

    positive_ic = (ic_df["ic"] > 0).mean()

    print(f"Rows              : {len(df):,}")
    print(f"RMSE              : {rmse:.6f}")
    print(f"R2                : {r2:.6f}")
    print(f"Pearson Corr      : {corr:.6f}")
    print(f"Mean Rank IC      : {mean_ic:.6f}")
    print(f"IC Std            : {ic_std:.6f}")
    print(f"IC IR             : {ic_ir:.6f}")
    print(f"Positive IC       : {positive_ic:.2%}")

    spread_df = top_bottom_spread(df)

    print()
    print("Top 20% vs Bottom 20%")
    print("-" * 70)

    if not spread_df.empty:

        mean_spread = spread_df["spread"].mean()
        spread_std = spread_df["spread"].std()
        positive_spread = (spread_df["spread"] > 0).mean()

        print(f"Trading days       : {len(spread_df)}")
        print(f"Mean Top-Bottom    : {mean_spread:.6f}")
        print(f"Spread Std         : {spread_std:.6f}")
        print(f"Positive Spread    : {positive_spread:.2%}")

    return {
        "rows": len(df),
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": corr,
        "mean_rank_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "positive_ic": positive_ic,
        "top_bottom_spread": (
            spread_df["spread"].mean()
            if not spread_df.empty
            else np.nan
        ),
    }


print("=" * 70)
print("LightGBM Alpha Model - Cross Sectional Rank Target")
print("=" * 70)


con = duckdb.connect(DB_PATH)

query = f"""
SELECT
    symbol,
    date,
    {", ".join(FEATURE_COLUMNS)},
    {TARGET_RAW}
FROM stock_features
WHERE {TARGET_RAW} IS NOT NULL
ORDER BY date, symbol
"""

df = con.execute(query).df()

con.close()

df["date"] = pd.to_datetime(df["date"])

print(f"Total rows    : {len(df):,}")
print(f"Symbols       : {df['symbol'].nunique()}")
print(
    f"Date range    : "
    f"{df['date'].min().date()} ~ {df['date'].max().date()}"
)


# ----------------------------------------------------------------------
# Cross-sectional target
# ----------------------------------------------------------------------

df = make_cross_sectional_rank(df)

df = df.dropna(
    subset=FEATURE_COLUMNS + ["target_rank_5d"]
).copy()


# ----------------------------------------------------------------------
# Time split
# ----------------------------------------------------------------------

train = df[
    df["date"] <= pd.Timestamp(TRAIN_END)
].copy()

validation = df[
    (df["date"] > pd.Timestamp(TRAIN_END))
    & (df["date"] <= pd.Timestamp(VALIDATION_END))
].copy()

test = df[
    df["date"] > pd.Timestamp(VALIDATION_END)
].copy()


print()
print("Data split")
print("-" * 70)

print(
    f"Train      : "
    f"{train['date'].min().date()} ~ "
    f"{train['date'].max().date()} "
    f"({len(train):,})"
)

print(
    f"Validation : "
    f"{validation['date'].min().date()} ~ "
    f"{validation['date'].max().date()} "
    f"({len(validation):,})"
)

print(
    f"Test       : "
    f"{test['date'].min().date()} ~ "
    f"{test['date'].max().date()} "
    f"({len(test):,})"
)


X_train = train[FEATURE_COLUMNS]
y_train = train["target_rank_5d"]

X_validation = validation[FEATURE_COLUMNS]
y_validation = validation["target_rank_5d"]

X_test = test[FEATURE_COLUMNS]
y_test = test["target_rank_5d"]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

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
print("-" * 70)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_validation, y_validation)
    ],
    callbacks=[
        __import__("lightgbm").early_stopping(
            stopping_rounds=100,
            verbose=True
        )
    ],
)


print()
print(f"Best iteration: {model.best_iteration_}")


# ----------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------

train["prediction"] = model.predict(X_train)
validation["prediction"] = model.predict(X_validation)
test["prediction"] = model.predict(X_test)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

train_metrics = print_metrics(
    "TRAIN",
    train
)

validation_metrics = print_metrics(
    "VALIDATION",
    validation
)

test_metrics = print_metrics(
    "TEST",
    test
)


# ----------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("=" * 70)
print("Feature Importance")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# ----------------------------------------------------------------------
# Save predictions
# ----------------------------------------------------------------------

prediction_df = pd.concat(
    [
        train[
            [
                "symbol",
                "date",
                TARGET_RAW,
                "target_rank_5d",
                "prediction",
            ]
        ],
        validation[
            [
                "symbol",
                "date",
                TARGET_RAW,
                "target_rank_5d",
                "prediction",
            ]
        ],
        test[
            [
                "symbol",
                "date",
                TARGET_RAW,
                "target_rank_5d",
                "prediction",
            ]
        ],
    ],
    ignore_index=True,
)


con = duckdb.connect(DB_PATH)

con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_rank_predictions"
)

con.register(
    "prediction_df",
    prediction_df
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_rank_predictions AS
    SELECT * FROM prediction_df
    """
)


con.execute(
    "DROP TABLE IF EXISTS alpha_lgbm_rank_importance"
)

con.register(
    "importance_df",
    importance
)

con.execute(
    """
    CREATE TABLE alpha_lgbm_rank_importance AS
    SELECT * FROM importance_df
    """
)


con.close()


print()
print("=" * 70)
print("Saved")
print("=" * 70)

print("alpha_lgbm_rank_predictions")
print("alpha_lgbm_rank_importance")