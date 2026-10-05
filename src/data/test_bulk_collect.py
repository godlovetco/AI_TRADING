import time
import sys

sys.path.append("src/data")

from toss_client import TossClient
from collect_history import collect_symbol, candles_to_dataframe


SYMBOLS = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "316140": "우리금융지주",
    "009150": "삼성전기",
    "402340": "SK스퀘어",
    "001820": "삼화콘덴서",
    "012330": "현대모비스",
    "034020": "두산에너빌리티",
    "105560": "KB금융",
    "042700": "한미반도체",
}


def main():
    client = TossClient()

    print("=" * 60)
    print("Toss API 대량 수집 속도 테스트")
    print("=" * 60)

    client.authenticate()

    results = []

    for symbol, name in SYMBOLS.items():

        print(f"\n[{symbol}] {name}")
        print("-" * 60)

        start = time.time()

        try:
            # collect_history.py의 기존 pagination 로직 사용
            original_name = None

            # collect_symbol()이 전역 SYMBOLS를 사용하므로
            # 임시로 이름을 주입
            import collect_history
            collect_history.SYMBOLS[symbol] = name

            df = collect_symbol(client, symbol)

            elapsed = time.time() - start

            results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "rows": len(df),
                    "seconds": elapsed,
                }
            )

            print()
            print(f"완료: {len(df)} rows")
            print(f"소요시간: {elapsed:.2f}초")

        except Exception as e:
            elapsed = time.time() - start

            print("ERROR:", e)

            results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "rows": 0,
                    "seconds": elapsed,
                }
            )

    print("\n")
    print("=" * 60)
    print("전체 테스트 결과")
    print("=" * 60)

    total_rows = 0
    total_seconds = 0

    for r in results:
        print(
            f"{r['symbol']} "
            f"{r['name']:12s} "
            f"{r['rows']:4d} rows "
            f"{r['seconds']:.2f} sec"
        )

        total_rows += r["rows"]
        total_seconds += r["seconds"]

    print("-" * 60)
    print(f"총 데이터: {total_rows:,} rows")
    print(f"총 시간: {total_seconds:.2f} sec")
    print(f"종목당 평균: {total_seconds / len(results):.2f} sec")
    print(f"종목당 평균 페이지 시간 기준 예상")
    print(f"100종목 예상: {total_seconds / len(results) * 10:.1f} sec")
    print(f"200종목 예상: {total_seconds / len(results) * 20:.1f} sec")


if __name__ == "__main__":
    main()
