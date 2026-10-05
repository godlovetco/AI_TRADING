import os
import requests
from dotenv import load_dotenv


class TossClient:
    BASE_URL = "https://openapi.tossinvest.com"

    def __init__(self):
        load_dotenv()

        self.client_id = os.getenv("TOSS_CLIENT_ID")
        self.client_secret = os.getenv("TOSS_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "TOSS_CLIENT_ID 또는 TOSS_CLIENT_SECRET이 없습니다."
            )

        self.access_token = None

    def authenticate(self):
        response = requests.post(
            f"{self.BASE_URL}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()
        self.access_token = data["access_token"]

        return data

    def _headers(self):
        if not self.access_token:
            self.authenticate()

        return {
            "Authorization": f"Bearer {self.access_token}"
        }

    def get_price(self, symbols):
        response = requests.get(
            f"{self.BASE_URL}/api/v1/prices",
            params={"symbols": symbols},
            headers=self._headers(),
            timeout=10,
        )

        response.raise_for_status()

        return response.json()["result"]

    def get_daily_candles(
        self,
        symbol,
        count=200,
        before=None,
        adjusted=True,
    ):
        params = {
            "symbol": symbol,
            "interval": "1d",
            "count": count,
            "adjusted": str(adjusted).lower(),
        }

        if before:
            params["before"] = before

        response = requests.get(
            f"{self.BASE_URL}/api/v1/candles",
            params=params,
            headers=self._headers(),
            timeout=10,
        )

        response.raise_for_status()

        return response.json()["result"]


if __name__ == "__main__":
    client = TossClient()

    price = client.get_price("005930")
    print("현재가:")
    print(price)

    candles = client.get_daily_candles("005930", count=5)

    print("\n최근 5개 일봉:")
    for candle in candles["candles"]:
        print(candle)

    print("\n다음 페이지 기준:")
    print(candles.get("nextBefore"))
