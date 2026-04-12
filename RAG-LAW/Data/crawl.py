import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://thuvienphapluat.vn/"
}


def save_preview_to_txt(law_data):
    """Ghi dữ liệu vừa trích xuất được ra file txt để kiểm tra nhanh"""
    file_name = "kiem_tra_ket_qua.txt"
    with open(file_name, "a", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"VĂN BẢN: {law_data['law_title']}\n")
        f.write(f"LINK: {law_data['url']}\n")
        f.write(f"TỔNG SỐ ĐIỀU: {law_data['num_articles']}\n")
        f.write("-" * 30 + "\n")
        # In thử 2 điều đầu tiên để xem content có bị trống hay không
        for i, art in enumerate(law_data['articles'][:2]):
            f.write(f"[{i+1}] {art['article']}\n")
            f.write(f"NỘI DUNG TRÍCH XUẤT:\n{art['content'][:1000]}\n") # Lấy 1000 ký tự đầu của content
            f.write("." * 20 + "\n")
        f.write("\n\n")

def get_html(url, retries=5):
    for i in range(retries):
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                return res.text
            elif res.status_code == 410 or res.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(random.uniform(2, 5))
    return None

def extract_articles(text):
    pattern = r"(^Điều\s+\d+[.\s][\s\S]*?)(?=\nĐiều\s+\d+[.\s]|$)"
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    
    articles = []
    for m in matches:
        lines = [line.strip() for line in m.split('\n') if line.strip()]
        if not lines: continue
        
        title = lines[0]
        if len(title) < 50 and len(lines) > 1:
            title = title + " " + lines[1]
        
        articles.append({
            "article": title,
            "content": m.strip()
        })
    return articles

def clean_text(html_content):
    text = re.sub(r'<style.*?>.*?</style>', '', html_content, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'<.*?>', '\n', text)
    text = re.sub(r'\n+', '\n', text).strip()
    return text

def parse_law(url):
    html = get_html(url)
    if not html: return None
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Lấy Tiêu đề văn bản
    title_tag = soup.select_one("h1")
    law_title = title_tag.get_text(strip=True) if title_tag else "Văn bản"

    # 2. Xác định vùng nội dung chính
    content_div = soup.select_one("#divContentDoc") or soup.select_one(".content1")
    if not content_div: return None

    # 3. LÀM SẠCH cực kỳ quan trọng
    for a in content_div.find_all("a"):
        a.unwrap() 
    
    for trash in content_div.select("script, style, .LinkVB, .note"):
        trash.decompose() # Xóa hẳn rác

    # 4. TRÍCH XUẤT NỘI DUNG (Duyệt theo thẻ để gom hàng)
    articles = []
    current_art = None
    
    # Tìm các thẻ block (chứa text cấp thấp nhất) để không bị lặp text giữa thẻ cha và thẻ con
    block_elements = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
    leaf_tags = []
    for tag in content_div.find_all(block_elements):
        # Nếu thẻ này không chứa thẻ block nào khác bên trong, nó là thẻ chứa nội dung cuối cùng
        if not tag.find(block_elements):
            leaf_tags.append(tag)
            
    for tag in leaf_tags:
        text = tag.get_text(separator=' ', strip=True)
        if not text: continue
        
        # Nhận diện Điều mới: Bắt đầu bằng chữ "Điều" + Số 
        if re.match(r'^Điều\s+\d+([\s\.\:\,]|$)', text, re.IGNORECASE):
            if current_art and len(current_art['content']) > 50: # Tránh lưu danh mục rác
                # Nếu độ dài của content gần bằng độ dài của title, flag as failed extraction
                if len(current_art['content']) <= len(current_art['article']) + 15:
                    print(f"Cảnh báo: Lỗi trích xuất '{current_art['article']}' (Chỉ có tiêu đề, thiếu nội dung).")
                articles.append(current_art)
            
            current_art = {
                "article": text, # Dòng tiêu đề Điều ban đầu
                "content": text, # Nội dung
            }
        else:
            if current_art:
                # Nếu chưa có dấu xuống dòng và độ dài Điều ngắn, có thể đó là phần tiếp theo của tiêu đề Điều
                if '\n' not in current_art['content'] and len(current_art['article']) < 50:
                    current_art['article'] += " " + text
                current_art['content'] += "\n" + text

    # Thêm điều cuối cùng
    if current_art and len(current_art['content']) > 50:
        if len(current_art['content']) <= len(current_art['article']) + 15:
            print(f"Cảnh báo: Lỗi trích xuất '{current_art['article']}' (Chỉ có tiêu đề, thiếu nội dung).")
        articles.append(current_art)

    if not articles: return None

    return {
        "law_title": law_title,
        "url": url,
        "num_articles": len(articles),
        "articles": articles
    }

def generate_target_urls():
    return [
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Hien-phap-nam-2013-215627.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Bo-luat-dan-su-2015-296215.aspx",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-hinh-su-2015-296661.aspx",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-dat-dai-2024-406180.aspx",
        "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-luat-lao-dong-2019-333670.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Doanh-nghiep-2020-435777.aspx",
        "https://thuvienphapluat.vn/van-ban/Vi-pham-hanh-chinh/Luat-Xu-ly-vi-pham-hanh-chinh-2012-143493.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Hon-nhan-va-gia-dinh-2014-239103.aspx",
        "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-Trat-tu-an-toan-giao-thong-duong-bo-2024-617835.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-Can-cuoc-2023-536412.aspx",
        "https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Bao-hiem-xa-hoi-2024-617830.aspx",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Nha-o-2023-533568.aspx",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Kinh-doanh-bat-dong-san-2023-529067.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Luat-Cac-to-chuc-tin-dung-2024-411330.aspx",
        "https://thuvienphapluat.vn/van-ban/Dau-thau-Cong-san/Luat-Dau-thau-2023-524435.aspx",
        "https://thuvienphapluat.vn/van-ban/Dau-tu/Luat-Dau-tu-2020-431835.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Luat-Giao-dich-dien-tu-2023-540153.aspx",
        "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-phi/Luat-Quan-ly-thue-2019-417387.aspx",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-To-tung-hinh-su-2015-296884.aspx",
        "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Bo-luat-to-tung-dan-su-2015-298344.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-ban-hanh-van-ban-quy-pham-phap-luat-2015-282300.aspx",
        "https://thuvienphapluat.vn/van-ban/Thuong-mai/Luat-Bao-ve-quyen-loi-nguoi-tieu-dung-2023-533564.aspx",
        "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Luat-Kham-benh-chua-benh-2023-535311.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-Cong-chung-2014-238299.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-Phong-chong-tham-nhung-2018-401800.aspx"
    ]

def get_dynamic_urls():
    dynamic_urls = set()
    urls_to_crawl = ["https://thuvienphapluat.vn/"]
    for url in urls_to_crawl:
        html = get_html(url)
        if not html: continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/van-ban/" in href and ".aspx" in href and "luat" in href.lower():
                if href.startswith("http"): dynamic_urls.add(href)
                else: dynamic_urls.add("https://thuvienphapluat.vn" + href)
    return list(dynamic_urls)

def main():
    # Xóa file cũ nếu có để ghi mới từ đầu
    if os.path.exists("kiem_tra_ket_qua.txt"):
        os.remove("kiem_tra_ket_qua.txt")

    target_urls = generate_target_urls()
    target_urls.extend(get_dynamic_urls())
    target_urls = list(dict.fromkeys(target_urls))
    random.shuffle(target_urls)
    
    all_data = []
    total_articles = 0
    valid_laws = 0
    seen = set()
    seen_titles = set()
    
    print(f"\nĐang bắt đầu crawl. Tổng số link cần rà soát: {len(target_urls)} links...")

    for i, link in enumerate(target_urls):
        if link in seen: continue
        seen.add(link)
        
        law = parse_law(link)
        if law and law['num_articles'] > 0:
            if law['law_title'] in seen_titles:
                print(f"[{i+1}/{len(target_urls)}] Đã có dữ liệu luật này, bỏ qua: {law['law_title']}")
                continue
                
            # Loại bỏ luật bị lấy sai theo cấu trúc
            if "Nghị định 91/2026/NĐ-CP" in law['law_title'] or "hướng dẫn Luật Giáo dục đại học" in law['law_title']:
                print(f"[{i+1}/{len(target_urls)}] Đã chặn luật cào sai: {law['law_title']}")
                continue
            
            seen_titles.add(law['law_title'])
            
            save_preview_to_txt(law)
            
            all_data.append(law)
            valid_laws += 1
            total_articles += law['num_articles']
            print(f"[{i+1}/{len(target_urls)}] Xong: {law['law_title']} (Các điều: {law['num_articles']})")
            
            flat_data = []
            for d_law in all_data:
                for art in d_law["articles"]:
                    flat_data.append({
                        "law_title": d_law["law_title"],
                        "article": art["article"],
                        "content": art["content"],
                        "url": d_law["url"]
                    })
            with open("law_articles_full.json", "w", encoding="utf-8") as f:
                json.dump(flat_data, f, ensure_ascii=False, indent=2)
        else:
            print(f"[{i+1}/{len(target_urls)}] Bỏ qua (không có nội dung phù hợp): {link}")
            
        time.sleep(random.uniform(1.5, 3.5))

if __name__ == "__main__":
    main()