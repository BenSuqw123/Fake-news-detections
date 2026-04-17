import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://thuvienphapluat.vn/",
    "Connection": "keep-alive"
}

TARGET_URLS = [
    "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Hien-phap-nam-2013-215627.aspx",
    "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Bo-luat-dan-su-2015-296215.aspx",
    "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Luat-An-ninh-mang-2025-so-116-2025-QH15-666020.aspx",
    "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-trat-tu-an-toan-giao-thong-duong-bo-2024-so-36-2024-QH15-444251.aspx",
    "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Can-cuoc-26-2023-QH15-552422.aspx",
    "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Luat-Thue-thu-nhap-ca-nhan-2025-so-109-2025-QH15-665870.aspx",
    "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Thue-thu-nhap-doanh-nghiep-2025-so-67-2025-QH15-580594.aspx",
    "https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Bao-hiem-xa-hoi-2024-557190.aspx",
    "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-333670.aspx",
    "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-Bao-ve-quyen-loi-nguoi-tieu-dung-2023-19-2023-QH15-500102.aspx",
    "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-Thuong-mai-dien-tu-2025-so-122-2025-QH15-662035.aspx"
]

def get_html_with_session(session, url):
    try:
        response = session.get(url, headers=HEADERS, timeout=25)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"   [!] Lỗi kết nối: {e}")
    return None

def parse_law(session, url):
    html = get_html_with_session(session, url)
    if not html: return None
    
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    law_title = title_tag.get_text(strip=True) if title_tag else "Văn bản"
    
    content_area = soup.find(id="divContentDoc") or soup.find(class_="content1")
    if not content_area: return None

    for trash in content_area.find_all(['script', 'style', 'iframe']):
        trash.decompose()

    articles = []
    current_art = None
    
    for element in content_area.find_all(['p', 'div', 'h2', 'h3', 'h4', 'li']):
        if element.name == 'div' and element.find(['p', 'div']): continue
            
        text = element.get_text(separator=' ', strip=True)
        if not text: continue

        if re.match(r'^(Điều|ĐIỀU)\s+\d+[\.\s\:\-]', text):
            if current_art: articles.append(current_art)
            current_art = {"law_title": law_title, "article": text, "content": text, "url": url}
        else:
            if current_art:
                if "\n" not in current_art['content'] and len(current_art['article']) < 80:
                    current_art['article'] += " " + text
                current_art['content'] += "\n" + text

    if current_art: articles.append(current_art)
    return articles

def main():
    output_json = r"D:\Fake-news-detections\RAG_LAW\Data\law_articles.json"
    output_report = r"D:\Fake-news-detections\RAG_LAW\Data\result.txt"
    all_data = []
    summary_list = []
    
    session = requests.Session()
    session.get("https://thuvienphapluat.vn/", headers=HEADERS)
    
    print(f"--- BẮT ĐẦU CÀO {len(TARGET_URLS)} VĂN BẢN ---")

    for i, url in enumerate(TARGET_URLS):
        print(f"[{i+1}/{len(TARGET_URLS)}] Đang xử lý: {url}")
        
        law_articles = parse_law(session, url)
        
        if law_articles:
            all_data.extend(law_articles)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            law_name = law_articles[0]['law_title']
            count = len(law_articles)
            summary_list.append((law_name, count))
            
            print(f"   =>{law_name} ({count} điều)")
        else:
            print(f"   => THẤT BẠI: {url}")
        
        time.sleep(random.uniform(4, 7))

    total_articles = sum(item[1] for item in summary_list)
    with open(output_report, "w", encoding="utf-8") as f_rep:
        f_rep.write("       BÁO CÁO TỔNG CỘNG DỮ LIỆU LUẬT      \n")
        f_rep.write("==========================================\n\n")
        f_rep.write(f"Tổng số văn bản đã cào: {len(summary_list)}/{len(TARGET_URLS)}\n")
        f_rep.write(f"Tổng số điều luật thu thập được: {total_articles} điều\n\n")
        f_rep.write("CHI TIẾT TỪNG VĂN BẢN:\n")
        f_rep.write("-" * 40 + "\n")
        for idx, (name, count) in enumerate(summary_list, 1):
            f_rep.write(f"{idx}. {name}\n")
            f_rep.write(f"   -> Số lượng: {count} điều\n")
            f_rep.write("-" * 40 + "\n")
            
    print(f"\n--- HOÀN TẤT ---")
    print(f"1. File dữ liệu: {output_json}")
    print(f"2. File tổng cộng: {output_report}")

if __name__ == "__main__":
    main()