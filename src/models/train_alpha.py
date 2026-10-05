import duckdb
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


DB_PATH = "data/market.duckdb"


# ============================================================
# Configuration
# ============================================================

TRAIN_END = "2024-12-31"
VALIDATION_END = "2025-12-31"

TOP_BOTTOM_PCT = 0.10

RIDGE_ALPHA = 10.0

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


# ============================================================
# Metrics
# ============================================================

def information_coefficient(df):
    """
    Cross-sectional Spearman Rank IC.

    Each date:
        predicted alpha ranking
        vs
        realized future excess return ranking
    """

    daily_ic = (
        df.groupby("date")
        .apply(
            lambda x: x["prediction"].corr(
                x[TARGET_COLUMN],
                method="spearman"
            ),
            include_groups=False,
        )
        .dropna()
    )

    return daily_ic


def max_drawdown(returns):
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()

    drawdown = cumulative / running_max - 1.0

    return drawdown.min()


def annualized_sharpe(returns):
    if len(returns) < 2:
        return np.nan

    std = returns.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return returns.mean() / std * np.sqrt(252)


# ============================================================
# Portfolio Backtest
# ============================================================

def long_short_backtest(df):

    daily_results = []

    for date, group in df.groupby("date"):

        group = group.dropna(
            subset=["prediction", TARGET_COLUMN]
        ).copy()

        if len(group) < 20:
            continue

        group = group.sort_values(
            "prediction",
            ascending=False
        )

        n = max(
            1,
            int(len(group) * TOP_BOTTOM_PCT)
        )

        long_group = group.head(n)
        short_group = group.tail(n)

        long_return = long_group[
            TARGET_COLUMN
        ].mean()

        short_return = short_group[
            TARGET_COLUMN
        ].mean()

        long_short_return = (
            long_return - short_return
        )

        daily_results.append(
            {
                "date": date,
                "long_return": long_return,
                "short_return": short_return,
                "long_short_return": long_short_return,
                "long_count": len(long_group),
                "short_count": len(short_group),
            }
        )

    return pd.DataFrame(daily_results)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ALPHA MODEL TRAINING")
    print("=" * 70)

    con = duckdb.connect(DB_PATH)

    # ========================================================
    # 1. Load data
    # ========================================================

    print()
    print("Loading stock_features...")

    columns = [
        "symbol",
        "date",
    ] + FEATURE_COLUMNS + [
        TARGET_COLUMN
    ]

    query = f"""
        SELECT
            {", ".join(columns)}
        FROM stock_features
        ORDER BY date, symbol
    """

    df = con.execute(query).fetchdf()

    df["date"] = pd.to_datetime(df["date"])

    print(f"Rows   : {len(df):,}")
    print(f"Symbols: {df['symbol'].nunique():,}")
    print(
        f"Period : "
        f"{df['date'].min().date()} ~ "
        f"{df['date'].max().date()}"
    )

    # ========================================================
    # 2. Time split
    # ========================================================

    train = df[
        df["date"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["date"] > TRAIN_END)
        & (df["date"] <= VALIDATION_END)
    ].copy()

    test = df[
        df["date"] > VALIDATION_END
    ].copy()

    print()
    print("=" * 70)
    print("TIME SPLIT")
    print("=" * 70)

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

    # ========================================================
    # 3. Prepare X / y
    # ========================================================

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # ========================================================
    # 4. Standardization
    # ========================================================

    print()
    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # 5. Ridge model
    # ========================================================

    print()
    print("Training Ridge Alpha model...")

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # 6. Predictions
    # ========================================================

    train["prediction"] = model.predict(
        X_train_scaled
    )

    validation["prediction"] = model.predict(
        X_validation_scaled
    )

    test["prediction"] = model.predict(
        X_test_scaled
    )

    # ========================================================
    # 7. Basic metrics
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL METRICS")
    print("=" * 70)

    for name, data, y_true in [
        ("Train", train, y_train),
        ("Validation", validation, y_validation),
        ("Test", test, y_test),
    ]:

        r2 = r2_score(
            y_true,
            data["prediction"]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                data["prediction"]
            )
        )

        corr = data[
            "prediction"
        ].corr(
            y_true,
            method="pearson"
        )

        print()
        print(name)

        print(f"  R²   : {r2:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  Corr : {corr:.6f}")

    # ========================================================
    # 8. Rank IC
    # ========================================================

    print()
    print("=" * 70)
    print("RANK IC")
    print("=" * 70)

    for name, data in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        ic = information_coefficient(data)

        print()
        print(name)

        print(f"  Mean Rank IC : {ic.mean():.6f}")
        print(f"  IC Std       : {ic.std():.6f}")

        if ic.std() != 0:
            print(
                f"  IC IR        : "
                f"{ic.mean() / ic.std() * np.sqrt(252):.6f}"
            )

        print(
            f"  Positive IC %: "
            f"{(ic > 0).mean():.2%}"
        )

    # ========================================================
    # 9. Long / Short Backtest
    # ========================================================

    print()
    print("=" * 70)
    print("LONG / SHORT BACKTEST")
    print("=" * 70)

    validation_bt = long_short_backtest(
        validation
    )

    test_bt = long_short_backtest(
        test
    )

    for name, bt in [
        ("Validation", validation_bt),
        ("Test", test_bt),
    ]:

        if bt.empty:
            continue

        returns = bt[
            "long_short_return"
        ]

        print()
        print(name)

        print(
            f"  Trading days : {len(bt):,}"
        )

        print(
            f"  Mean daily LS: "
            f"{returns.mean():.6f}"
        )

        print(
            f"  Annual return: "
            f"{returns.mean() * 252:.2%}"
        )

        print(
            f"  Annual vol   : "
            f"{returns.std() * np.sqrt(252):.2%}"
        )

        print(
            f"  Sharpe       : "
            f"{annualized_sharpe(returns):.4f}"
        )

        print(
            f"  Max Drawdown : "
            f"{max_drawdown(returns):.2%}"
        )

    # ========================================================
    # 10. Save predictions
    # ========================================================

    print()
    print("Saving predictions...")

    all_predictions = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )

    prediction_columns = [
        "symbol",
        "date",
        "prediction",
        TARGET_COLUMN,
        "split",
    ]

    prediction_df = all_predictions[
        prediction_columns
    ].copy()

    con.execute(
        "DROP TABLE IF EXISTS alpha_predictions"
    )

    con.register(
        "prediction_df",
        prediction_df
    )

    con.execute("""
        CREATE TABLE alpha_predictions AS
        SELECT *
        FROM prediction_df
    """)

    con.unregister(
        "prediction_df"
    )

    # ========================================================
    # 11. Save backtest results
    # ========================================================

    con.execute(
        "DROP TABLE IF EXISTS alpha_backtest"
    )

    backtest_df = pd.concat(
        [
            validation_bt.assign(
                split="validation"
            ),
            test_bt.assign(
                split="test"
            ),
        ],
        ignore_index=True,
    )

    con.register(
        "backtest_df",
        backtest_df
    )

    con.execute("""
        CREATE TABLE alpha_backtest AS
        SELECT *
        FROM backtest_df
    """)

    con.unregister(
        "backtest_df"
    )

    # ========================================================
    # 12. Model coefficients
    # ========================================================

    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": model.coef_,
        }
    )

    coefficients[
        "abs_coefficient"
    ] = coefficients[
        "coefficient"
    ].abs()

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print()
    print("=" * 70)
    print("FEATURE COEFFICIENTS")
    print("=" * 70)

    print(
        coefficients.to_string(
            index=False
        )
    )

    con.register(
        "coefficients_df",
        coefficients
    )

    con.execute(
        "DROP TABLE IF EXISTS alpha_coefficients"
    )

    con.execute("""
        CREATE TABLE alpha_coefficients AS
        SELECT *
        FROM coefficients_df
    """)

    con.unregister(
        "coefficients_df"
    )

    con.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()