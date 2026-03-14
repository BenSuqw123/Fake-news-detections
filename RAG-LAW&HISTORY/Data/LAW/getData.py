import requests
import json
import time
import os

API_KEY = "68eb5e3536be7ca5d26dfbf3b3345c2d793d01b1"

headers = {
    "Authorization": f"Token {API_KEY}",
    "User-Agent": "Mozilla/5.0"
}

url = "https://www.courtlistener.com/api/rest/v4/opinions/"
save_file = "scotus_cases.json"

# load dữ liệu cũ nếu có
if os.path.exists(save_file):
    with open(save_file, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
else:
    all_cases = []

print("Existing cases:", len(all_cases))

while url:
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

    except Exception as e:
        print("Error:", e)
        print("Retry after 10 seconds...")
        time.sleep(10)
        continue

    for case in data.get("results", []):
        case_data = {
            "case_name": case.get("case_name"),
            "date_filed": case.get("date_filed"),
            "text": case.get("plain_text") or ""
        }

        all_cases.append(case_data)

    print("Downloaded:", len(all_cases))

    # SAVE NGAY SAU MỖI PAGE
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False)

    url = data.get("next")

    time.sleep(2)  # tránh rate limit

print("Done!")