import duckdb
import pandas as pd
import numpy as np


DB_PATH = "data/market.duckdb"


# ============================================================
# Technical Indicators
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    return atr


# ============================================================
# Stock Features
# ============================================================

def build_stock_features(df):
    df = df.sort_values("date").copy()

    close = df["close"]
    volume = df["volume"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    # --------------------------------------------------------
    # Moving Average
    # --------------------------------------------------------

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    df["ma5_ratio"] = close / ma5 - 1
    df["ma20_ratio"] = close / ma20 - 1
    df["ma60_ratio"] = close / ma60 - 1
    df["ma120_ratio"] = close / ma120 - 1

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_20d"] = (
        df["ret_1d"]
        .rolling(20)
        .std()
    )

    df["volatility_60d"] = (
        df["ret_1d"]
        .rolling(60)
        .std()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_ma20 = volume.rolling(20).mean()

    df["volume_ratio_20d"] = (
        volume / volume_ma20
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        close,
        period=14
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr_14"] = calculate_atr(
        df,
        period=14
    )

    # ATR relative to price
    df["atr_ratio_14"] = (
        df["atr_14"] / close
    )

    return df


# ============================================================
# Main
# ============================================================

def main():

    con = duckdb.connect(DB_PATH)

    print("=" * 70)
    print("200-STOCK FEATURE ENGINEERING")
    print("=" * 70)

    # ========================================================
    # 1. Load ALL stock data
    # ========================================================

    prices = con.execute("""
        SELECT
            symbol,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM daily_prices
        ORDER BY symbol, date
    """).df()

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    print(f"Raw rows: {len(prices):,}")
    print(
        f"Symbols: {prices['symbol'].nunique():,}"
    )
    print(
        f"Date range: "
        f"{prices['date'].min().date()} ~ "
        f"{prices['date'].max().date()}"
    )

    # ========================================================
    # 2. Build stock technical features
    # ========================================================

    print()
    print("Building stock features...")

    feature_list = []

    for symbol, group in prices.groupby("symbol", sort=False):

        feature_df = build_stock_features(group)

        feature_list.append(feature_df)

    features = pd.concat(
        feature_list,
        ignore_index=True
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    # ========================================================
    # 3. Load KOSPI
    # ========================================================

    print()
    print("Loading KOSPI data...")

    kospi = con.execute("""
        SELECT
            date,
            close
        FROM kospi_daily
        ORDER BY date
    """).df()

    kospi["date"] = pd.to_datetime(
        kospi["date"]
    )

    kospi = kospi.sort_values("date").copy()

    # ========================================================
    # 4. KOSPI market features
    # ========================================================

    kospi["kospi_return_1d"] = (
        kospi["close"].pct_change(1)
    )

    kospi["kospi_return_5d"] = (
        kospi["close"].pct_change(5)
    )

    kospi["kospi_return_20d"] = (
        kospi["close"].pct_change(20)
    )

    kospi["kospi_return_60d"] = (
        kospi["close"].pct_change(60)
    )

    # ========================================================
    # 5. Future KOSPI return
    # ========================================================

    kospi["future_kospi_return_5d"] = (
        kospi["close"].shift(-5)
        / kospi["close"]
        - 1
    )

    # ========================================================
    # 6. Merge market data
    # ========================================================

    features = features.merge(
        kospi[
            [
                "date",
                "kospi_return_1d",
                "kospi_return_5d",
                "kospi_return_20d",
                "kospi_return_60d",
                "future_kospi_return_5d",
            ]
        ],
        on="date",
        how="left",
    )

    # ========================================================
    # 7. Cross-sectional relative strength
    # ========================================================

    features["relative_strength_5d"] = (
        features["ret_5d"]
        - features["kospi_return_5d"]
    )

    features["relative_strength_20d"] = (
        features["ret_20d"]
        - features["kospi_return_20d"]
    )

    features["relative_strength_60d"] = (
        features["ret_60d"]
        - features["kospi_return_60d"]
    )

    # ========================================================
    # 8. Future stock return
    # ========================================================

    features = features.sort_values(
        ["symbol", "date"]
    ).copy()

    features["future_stock_return_5d"] = (
        features
        .groupby("symbol")["close"]
        .shift(-5)
        / features["close"]
        - 1
    )

    # ========================================================
    # 9. Future excess return
    #
    # Alpha model target
    #
    # Stock future return
    #      -
    # KOSPI future return
    # ========================================================

    features["future_excess_return_5d"] = (
        features["future_stock_return_5d"]
        - features["future_kospi_return_5d"]
    )

    # ========================================================
    # 10. Feature columns
    # ========================================================

    feature_columns = [
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

    # ========================================================
    # 11. Remove rows without usable target/features
    # ========================================================

    required_columns = (
        feature_columns
        + ["future_excess_return_5d"]
    )

    before_drop = len(features)

    features_clean = features.dropna(
        subset=required_columns
    ).copy()

    after_drop = len(features_clean)

    print()
    print(
        f"Rows before NaN filtering: "
        f"{before_drop:,}"
    )

    print(
        f"Rows after NaN filtering:  "
        f"{after_drop:,}"
    )

    print(
        f"Removed rows: "
        f"{before_drop - after_drop:,}"
    )

    # ========================================================
    # 12. Save to DuckDB
    # ========================================================

    con.execute("""
        DROP TABLE IF EXISTS stock_features
    """)

    con.register(
        "features_df",
        features_clean
    )

    con.execute("""
        CREATE TABLE stock_features AS
        SELECT *
        FROM features_df
    """)

    con.unregister(
        "features_df"
    )

    # ========================================================
    # 13. Integrity check
    # ========================================================

    result = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT symbol) AS symbols,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM stock_features
    """).fetchone()

    print()
    print("=" * 70)
    print("FEATURE TABLE")
    print("=" * 70)

    print(f"Rows   : {result[0]:,}")
    print(f"Symbols: {result[1]:,}")
    print(f"Period : {result[2]} ~ {result[3]}")

    # ========================================================
    # 14. Per-symbol row count
    # ========================================================

    symbol_stats = con.execute("""
        SELECT
            COUNT(*) AS symbol_count,
            MIN(row_count) AS min_rows,
            MAX(row_count) AS max_rows,
            AVG(row_count) AS avg_rows
        FROM (
            SELECT
                symbol,
                COUNT(*) AS row_count
            FROM stock_features
            GROUP BY symbol
        )
    """).fetchone()

    print()
    print("Per-symbol statistics:")
    print(
        f"  Symbols : {symbol_stats[0]:,}"
    )
    print(
        f"  Min rows: {symbol_stats[1]:,}"
    )
    print(
        f"  Max rows: {symbol_stats[2]:,}"
    )
    print(
        f"  Avg rows: {symbol_stats[3]:,.1f}"
    )

    # ========================================================
    # 15. Feature NaN check
    # ========================================================

    print()
    print("Feature NaN check:")

    for column in feature_columns:
        null_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM stock_features
            WHERE "{column}" IS NULL
            """
        ).fetchone()[0]

        print(
            f"  {column:30s}: {null_count:,}"
        )

    # ========================================================
    # 16. Target statistics
    # ========================================================

    target_stats = con.execute("""
        SELECT
            AVG(future_excess_return_5d),
            STDDEV(future_excess_return_5d),
            MIN(future_excess_return_5d),
            MAX(future_excess_return_5d)
        FROM stock_features
    """).fetchone()

    print()
    print("Target statistics:")
    print(
        f"  Mean : {target_stats[0]:.6f}"
    )
    print(
        f"  Std  : {target_stats[1]:.6f}"
    )
    print(
        f"  Min  : {target_stats[2]:.6f}"
    )
    print(
        f"  Max  : {target_stats[3]:.6f}"
    )

    # ========================================================
    # 17. Final schema
    # ========================================================

    print()
    print("Final columns:")

    columns = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'stock_features'
        ORDER BY ordinal_position
    """).fetchall()

    for i, (column,) in enumerate(columns, start=1):
        print(
            f"{i:2d}. {column}"
        )

    con.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()