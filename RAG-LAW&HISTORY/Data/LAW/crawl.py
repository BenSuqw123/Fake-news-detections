import requests
import json
import time
import os
import re

# --- CẤU HÌNH ---
API_KEY = "68eb5e3536be7ca5d26dfbf3b3345c2d793d01b1"
TARGET_COUNT = 6000
SAVE_FILE = "courtlistener_6000_data.jsonl"
HEADERS = {"Authorization": f"Token {API_KEY}", "User-Agent": "LegalScraper/2.0"}

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\f', ' ', text) # Bỏ phân trang
    text = re.sub(r'\s+', ' ', text) # Gom khoảng trắng
    return text.strip()

# Kiểm tra tiến độ cũ để cào tiếp nếu bị ngắt quãng
current_count = 0
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "r", encoding="utf-8") as f:
        current_count = sum(1 for line in f)

print(f"📈 Tiến độ hiện tại: {current_count}/{TARGET_COUNT}")

# Bắt đầu từ trang đầu tiên hoặc bạn có thể chỉ định trang tiếp theo
url = "https://www.courtlistener.com/api/rest/v4/opinions/?page=1"

with open(SAVE_FILE, "a", encoding="utf-8") as f_out:
    while url and current_count < TARGET_COUNT:
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            if response.status_code == 429: # Chạm ngưỡng giới hạn (Rate Limit)
                print("⏳ Chạm giới hạn API. Nghỉ 60s...")
                time.sleep(60)
                continue
                
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            for item in results:
                if current_count >= TARGET_COUNT:
                    break
                
                # Trích xuất văn bản
                content = item.get("plain_text") or ""
                if not content and item.get("html_with_citations"):
                    content = re.sub('<[^<]+?>', '', item.get("html_with_citations"))
                
                if content.strip():
                    # Tạo tiêu đề từ URL slug
                    slug = item.get("absolute_url", "").split('/')[-2]
                    title = slug.replace('-', ' ').title()
                    
                    record = {
                        "id": item.get("id"),
                        "title_en": title,
                        "content_en": clean_text(content),
                        "date_filed": item.get("date_filed"),
                        "url": f"https://www.courtlistener.com{item.get('absolute_url')}",
                        "source": "CourtListener"
                    }
                    
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    current_count += 1

            print(f"✅ Đã lưu: {current_count}/{TARGET_COUNT} - Đang ở trang: {url}")
            
            url = data.get("next")
            time.sleep(0.5) # Nghỉ ngắn giữa các trang để an toàn

        except Exception as e:
            print(f"❌ Lỗi: {e}. Thử lại sau 5s...")
            time.sleep(5)
            continue

print(f"🏁 Hoàn thành! File lưu tại: {SAVE_FILE}")