import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/market.duckdb"


def evaluate_strategy(df, score_column, ascending=False):
    rows = []

    for date, group in df.groupby("date"):

        group = group.dropna(
            subset=[score_column, "target_excess_ret_5d"]
        )

        if len(group) < 2:
            continue

        group = group.sort_values(
            score_column,
            ascending=ascending
        )

        top = group.iloc[0]
        bottom = group.iloc[-1]

        rows.append({
            "date": date,
            "top_return": top["target_excess_ret_5d"],
            "bottom_return": bottom["target_excess_ret_5d"]
        })

    result = pd.DataFrame(rows)

    return {
        "days": len(result),
        "top_mean": result["top_return"].mean(),
        "bottom_mean": result["bottom_return"].mean(),
        "top_positive": (
            result["top_return"] > 0
        ).mean(),
        "bottom_positive": (
            result["bottom_return"] > 0
        ).mean(),
    }


def main():

    print("=" * 70)
    print("Baseline Alpha Test")
    print("=" * 70)

    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT
            symbol,
            date,
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
        WHERE date >= '2025-07-01'
        ORDER BY date, symbol
    """).fetchdf()

    strategies = {

        "RET_5D": {
            "column": "ret_5d",
            "ascending": False
        },

        "RET_20D": {
            "column": "ret_20d",
            "ascending": False
        },

        "MA60": {
            "column": "ma60_ratio",
            "ascending": False
        },

        "RSI": {
            "column": "rsi_14",
            "ascending": False
        },

        "VOLATILITY": {
            "column": "volatility_20d",
            "ascending": True
        },
    }

    results = []

    for name, config in strategies.items():

        result = evaluate_strategy(
            df,
            config["column"],
            config["ascending"]
        )

        results.append({
            "strategy": name,
            **result
        })

    result_df = pd.DataFrame(results)

    print()
    print(result_df.to_string(index=False))

    print()
    print("=" * 70)
    print("Random Baseline")
    print("=" * 70)

    random_results = []

    rng = np.random.default_rng(42)

    for date, group in df.groupby("date"):

        if len(group) < 2:
            continue

        returns = group[
            "target_excess_ret_5d"
        ].values

        random_results.append(
            rng.choice(returns)
        )

    random_results = np.array(random_results)

    print(
        f"days            : {len(random_results)}"
    )

    print(
        f"mean return     : "
        f"{random_results.mean():.6f}"
    )

    print(
        f"positive ratio  : "
        f"{(random_results > 0).mean():.2%}"
    )

    con.close()

    print()
    print("=" * 70)
    print("Baseline Test 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
