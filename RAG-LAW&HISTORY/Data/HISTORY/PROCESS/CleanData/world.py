import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import os
import random

# --- CẤU HÌNH HỆ THỐNG ---
CONFIG = {
    "base_path": r'D:\Fake-news-detections\RAG\HISTORY',
    "output_file": 'world_history_gold_2000.jsonl',
    "target_limit": 2000,
    "chrome_version": 145, # Khớp với phiên bản máy bạn
    "categories": [1, 2]   # 1: Articles, 2: Biographies
}

def setup_driver():
    """Khởi tạo trình duyệt chống bị phát hiện."""
    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')
    # Giữ nguyên cấu hình của bạn
    driver = uc.Chrome(options=options, version_main=CONFIG["chrome_version"])
    return driver

def save_to_jsonl(data, file_path):
    """Lưu dữ liệu theo định dạng JSONL (tiết kiệm bộ nhớ)."""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    if not os.path.exists(CONFIG["base_path"]): 
        os.makedirs(CONFIG["base_path"])
    
    full_output_path = os.path.join(CONFIG["base_path"], CONFIG["output_file"])
    driver = setup_driver()
    links = set()
    
    try:
        # BƯỚC 1: QUÉT DANH SÁCH LINK
        print("🔍 Đang thu thập danh sách bài viết...")
        for cat in CONFIG["categories"]:
            page = 1
            while len(links) < CONFIG["target_limit"]:
                try:
                    driver.get(f"https://www.worldhistory.org/type/{cat}/{page}/")
                except:
                    print("⚠️ Trình duyệt bị mất kết nối, đang thử lại...")
                    break
                
                # Đợi trang tải xong
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/article/'], a[href*='/definition/']"))
                    )
                except:
                    print("⚠️ Cloudflare hoặc hết trang. Hãy kiểm tra trình duyệt!")
                    break

                elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/article/'], a[href*='/definition/']")
                new_added = 0
                for e in elements:
                    try:
                        href = e.get_attribute("href")
                        if href and "/type/" not in href and href not in links:
                            links.add(href)
                            new_added += 1
                        if len(links) >= CONFIG["target_limit"]: break
                    except:
                        continue
                
                print(f"✅ Đã gom: {len(links)}/{CONFIG['target_limit']} link (Trang {page})")
                if new_added == 0: break
                page += 1
                time.sleep(random.uniform(2, 4))

        # BƯỚC 2: TRÍCH XUẤT NỘI DUNG CHI TIẾT
        print(f"\n🚀 Bắt đầu cào {len(links)} bài viết...")
        count = 0
        for link in list(links):
            try:
                driver.get(link)
                time.sleep(random.uniform(4, 6)) # Tránh bị chặn IP
                
                title = driver.find_element(By.TAG_NAME, "h1").text
                # Lấy nội dung trong thẻ article
                content = driver.find_element(By.TAG_NAME, "article").text
                
                if len(content) > 500:
                    record = {
                        "id": count + 1,
                        "title": title,
                        "content": content[:12000], # Giới hạn độ dài tối ưu cho AI
                        "url": link,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_to_jsonl(record, full_output_path)
                    count += 1
                    print(f"📦 [{count}] Đã lưu: {title[:45]}...")
                
                if count >= CONFIG["target_limit"]: break
            except Exception as e:
                # Nếu trình duyệt bị đóng tay, thoát vòng lặp để dọn dẹp
                if "no such window" in str(e).lower():
                    break
                print(f"❌ Lỗi tại {link}: {str(e)[:50]}")
                continue

    finally:
        # PHẦN QUAN TRỌNG: Sửa lỗi WinError 6 tại đây
        print("\n🏁 Hoàn thành! Đang dọn dẹp hệ thống...")
        try:
            # Chỉ gọi quit khi driver còn tồn tại và cửa sổ còn mở
            if driver and hasattr(driver, 'window_handles') and len(driver.window_handles) > 0:
                driver.quit()
        except:
            # Nếu vẫn lỗi Handle, bỏ qua luôn để tránh hiện Traceback
            pass

if __name__ == "__main__":
    main()