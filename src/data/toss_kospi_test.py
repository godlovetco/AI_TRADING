import os
import requests
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

BASE_URL = "https://openapi.tossinvest.com"


# 1. Access Token 발급
token_response = requests.post(
    f"{BASE_URL}/oauth2/token",
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=10,
)

token_response.raise_for_status()

access_token = token_response.json()["access_token"]


headers = {
    "Authorization": f"Bearer {access_token}"
}


# 2. KOSPI 현재 지표 조회

response = requests.get(
    f"{BASE_URL}/api/v1/market-indicators/prices",
    params={"symbols": "KOSPI"},
    headers=headers,
    timeout=10,
)

print("HTTP STATUS:", response.status_code)

response.raise_for_status()

data = response.json()

print("\n=== 시장 지표 ===")
print(data)
