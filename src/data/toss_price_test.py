import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

token_response = requests.post(
    "https://openapi.tossinvest.com/oauth2/token",
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=10,
)

token_response.raise_for_status()
access_token = token_response.json()["access_token"]

response = requests.get(
    "https://openapi.tossinvest.com/api/v1/prices",
    params={"symbols": "005930"},
    headers={
        "Authorization": f"Bearer {access_token}"
    },
    timeout=10,
)

print("HTTP STATUS:", response.status_code)

response.raise_for_status()

data = response.json()

print("삼성전자 API 조회 성공")
print(data)

