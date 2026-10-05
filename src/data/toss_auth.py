import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("TOSS_CLIENT_ID")
CLIENT_SECRET = os.getenv("TOSS_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("TOSS_CLIENT_ID 또는 TOSS_CLIENT_SECRET이 없습니다.")

url = "https://openapi.tossinvest.com/oauth2/token"

response = requests.post(
    url,
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=10,
)

print("HTTP STATUS:", response.status_code)

if response.ok:
    data = response.json()
    token = data.get("access_token")

    if token:
        print("TOSS API AUTH: OK")
        print("ACCESS TOKEN: 발급 성공")
        print("TOKEN TYPE:", data.get("token_type"))
        print("EXPIRES IN:", data.get("expires_in"))
    else:
        print("응답에 access_token이 없습니다.")
        print(response.text)
else:
    print("TOSS API AUTH: FAILED")
    print(response.text)
