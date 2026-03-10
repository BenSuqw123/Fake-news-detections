import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import os
import random

# --- CẤU HÌNH ---
CONFIG = {
    "base_path": r'D:\Fake-news-detections\RAG\HISTORY',
    "output_file": 'world_history_gold_2000.jsonl',
    "target_limit": 2000,
    "chrome_version": 145, 
}

def setup_driver():
    """Khởi tạo trình duyệt với User-Agent thật để tránh Cloudflare."""
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-popup-blocking')
    # Giả lập User-Agent trình duyệt thật
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
    
    driver = uc.Chrome(options=options, version_main=CONFIG["chrome_version"])
    return driver

def clean_ads_and_junk(driver):
    """Xóa bỏ các thành phần rác trên trang web trước khi trích xuất."""
    driver.execute_script("""
        const junkSelectors = [
            'nav', 'header', 'footer', 'aside', '.adsbygoogle', 
            '.promo-box', '.social-share', '#div-gpt-ad', 
            '.print-hide', 'script', 'style', '.sidebar', '.ad-slot'
        ];
        junkSelectors.forEach(s => {
            document.querySelectorAll(s).forEach(el => el.remove());
        });
    """)

def main():
    # 1. Khởi tạo thư mục
    if not os.path.exists(CONFIG["base_path"]): 
        os.makedirs(CONFIG["base_path"])
    
    full_output_path = os.path.join(CONFIG["base_path"], CONFIG["output_file"])
    
    # 2. Khởi tạo trình duyệt
    driver = setup_driver()
    links = set()
    
    try:
        # BƯỚC 1: LẤY DANH SÁCH LINK (Vượt Cloudflare bằng cách vào trang chủ trước)
        print("🌐 Đang kết nối tới trang chủ để làm ấm trình duyệt...")
        driver.get("https://www.worldhistory.org")
        time.sleep(random.uniform(5, 8))

        categories = [1, 2] # 1: Articles, 2: Biographies
        for cat in categories:
            page = 1
            while len(links) < CONFIG["target_limit"]:
                print(f"🔎 Đang quét Category {cat}, Trang {page}...")
                driver.get(f"https://www.worldhistory.org/type/{cat}/{page}/")
                
                try:
                    # Chờ tối đa 15 giây để nội dung bài viết load
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/article/'], a[href*='/definition/']"))
                    )
                except:
                    print("⚠️ Không tìm thấy link. Nếu thấy Cloudflare 'Verify you are human', hãy click vào nó!")
                    time.sleep(10) # Dành thời gian cho bạn giải captcha nếu kẹt
                    
                    # Thử kiểm tra lại sau khi đợi
                    elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/article/'], a[href*='/definition/']")
                    if not elements: break

                elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/article/'], a[href*='/definition/']")
                found_on_page = 0
                for e in elements:
                    href = e.get_attribute("href")
                    # Lọc link bài viết, loại bỏ link chuyển trang
                    if href and any(x in href for x in ['/article/', '/definition/']) and "/type/" not in href:
                        if href not in links:
                            links.add(href)
                            found_on_page += 1
                    if len(links) >= CONFIG["target_limit"]: break
                
                print(f"✅ Đã thu thập: {len(links)}/{CONFIG['target_limit']} link")
                if found_on_page == 0: break
                page += 1
                time.sleep(random.uniform(3, 5))

        # BƯỚC 2: TRÍCH XUẤT NỘI DUNG SẠCH
        print(f"\n🚀 Bắt đầu cào {len(links)} bài viết...")
        count = 0
        for link in list(links):
            try:
                driver.get(link)
                # Chờ nội dung chính bài viết hiện ra
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                
                # Cuộn trang để các thành phần được load đầy đủ
                driver.execute_script("window.scrollTo(0, 600);")
                time.sleep(random.uniform(3, 5))
                
                # Xóa rác và quảng cáo
                clean_ads_and_junk(driver)
                
                # Lấy tiêu đề và nội dung tinh khiết (chỉ thẻ p)
                title = driver.find_element(By.TAG_NAME, "h1").text
                p_elements = driver.find_elements(By.CSS_SELECTOR, "article p")
                content = "\n\n".join([p.text for p in p_elements if len(p.text.strip()) > 50])
                
                if len(content) > 500:
                    record = {
                        "id": count + 1,
                        "title": title,
                        "content": content,
                        "url": link,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Lưu luôn vào file (đề phòng crash)
                    with open(full_output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    count += 1
                    print(f"📦 [{count}] Đã lưu: {title[:40]}... ({len(content)} ký tự)")
                
                if count >= CONFIG["target_limit"]: break

            except Exception as e:
                if "no such window" in str(e).lower(): break
                print(f"❌ Lỗi tại {link}: {str(e)[:50]}")
                continue

    except Exception as e:
        print(f"💥 Lỗi hệ thống: {e}")
    finally:
        print("\n🏁 Đã hoàn thành hoặc dừng lại. Đang dọn dẹp trình duyệt...")
        try:
            driver.quit()
        except:
            pass

# --- ĐIỂM KHỞI CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    main()