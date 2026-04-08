import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://thuvienphapluat.vn/"
}

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
    pattern = r"(Điều\s+\d+[\.\:\s][\s\S]*?)(?=\nĐiều\s+\d+[\.\:\s]|$)"
    matches = re.findall(pattern, text)
    articles = []
    for m in matches:
        lines = [line.strip() for line in m.split('\n') if line.strip()]
        title = ""
        if lines:
            title = lines[0]
            if len(title) < 25 and len(lines) > 1:
                title = title + " " + lines[1]
        
        articles.append({
            "article": title if title else "Điều",
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
    # Cải thiện lấy Title (Law Title)
    title_tag = soup.select_one("h1")
    title = ""
    if title_tag and title_tag.text.strip():
        title = title_tag.text.strip()
    else:
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()
        elif soup.title:
            title = soup.title.text.replace(" - thuvienphapluat.vn", "").strip()
        else:
            title = url.split('/')[-1]

    content_div = (
        soup.select_one("#divContentDoc") or 
        soup.select_one(".content1") or 
        soup.select_one("article") or
        soup.select_one(".doc-content") or
        soup.select_one("#ctl00_plcMain_divContent") or
        soup.select_one(".noi-dung") or
        soup.find("div", class_=re.compile(r"content", re.I))
    )

    if not content_div: return None

    # Khắc phục lỗi lặp dữ liệu do thẻ HTML lồng nhau (VD: div chứa p chứa span)
    text = content_div.get_text(separator='\n', strip=True)
        
    text = re.sub(r'\n+', '\n', text).strip()

    if len(text) < 1000: return None

    articles = extract_articles(text)

    if not articles:
        pattern = r"(Điều\s+\d+[\.\:\s].+?(?=Điều\s+\d+[\.\:\s]|$))"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for m in matches:
            lines = [line.strip() for line in m.split('\n') if line.strip()]
            title = lines[0] if lines else "Điều"
            if len(title) < 25 and len(lines) > 1:
                title = title + " " + lines[1]
            articles.append({
                "article": title,
                "content": m.strip()
            })

    if not articles: return None

    return {
        "law_title": title,
        "url": url,
        "num_articles": len(articles),
        "articles": articles
    }

def generate_target_urls():
    return [
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-hinh-su-2015-296661.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Bo-luat-dan-su-2015-296215.aspx",
        "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-luat-lao-dong-2019-333670.aspx",
        "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-dat-dai-2024-406180.aspx",
        "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Luat-Doanh-nghiep-2020-435777.aspx",
        "https://thuvienphapluat.vn/van-ban/Dau-tu/Luat-Dau-tu-2020-431835.aspx",
        "https://thuvienphapluat.vn/van-ban/Vi-pham-hanh-chinh/Luat-Xu-ly-vi-pham-hanh-chinh-2012-143493.aspx",
        "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Luat-Kham-benh-chua-benh-2023-535311.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Hien-phap-nam-2013-215627.aspx",
        "https://thuvienphapluat.vn/van-ban/Bao-hiem/Luat-Bao-hiem-xa-hoi-2014-259700.aspx",
        "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-Giao-thong-duong-bo-2008-23-2008-QH12-82404.aspx",
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-To-tung-hinh-su-2015-296884.aspx",
        "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Bo-luat-to-tung-dan-su-2015-298344.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Luat-Thuong-mai-2005-36-2005-QH11-53232.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-nguyen-Moi-truong/Luat-Bao-ve-moi-truong-2020-436154.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-Can-cuoc-2023-536412.aspx",
        "https://thuvienphapluat.vn/van-ban/Giao-duc/Luat-Giao-duc-2019-417409.aspx",
        "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Luat-Cac-to-chuc-tin-dung-2024-411330.aspx",
        "https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Hon-nhan-va-gia-dinh-2014-239103.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-Cu-tru-2020-458117.aspx",
        "https://thuvienphapluat.vn/van-ban/Quoc-phong-An-ninh/Luat-Cong-an-nhan-dan-2018-386007.aspx",
        "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Luat-to-tung-hanh-chinh-2015-296538.aspx",
        "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-To-chuc-chinh-phu-2015-282384.aspx"
    ]

def get_dynamic_urls():
    dynamic_urls = set()
    urls_to_crawl = [
        "https://thuvienphapluat.vn/",
        "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/"
    ]
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
    target_urls = generate_target_urls()
    target_urls.extend(get_dynamic_urls())
    target_urls = list(dict.fromkeys(target_urls))
    random.shuffle(target_urls)
    
    all_data = []
    total_articles = 0
    valid_laws = 0
    seen = set()
    seen_titles = set()
    
    print(f"\n🚀 Đang bắt đầu crawl. Tổng số link cần rà soát: {len(target_urls)} links...")

    for i, link in enumerate(target_urls):
        # Bỏ giới hạn, cho phép crawl toàn bộ các link đã thu thập được
        # if valid_laws >= 10 and total_articles >= 200:
        #     break
            
        if link in seen: continue
        seen.add(link)
        
        law = parse_law(link)
        if law and law['num_articles'] > 0:
            # Ngăn chặn việc cùng 1 luật nhưng bị crawl nhiều lần do URL khác nhau
            if law['law_title'] in seen_titles:
                print(f"[{i+1}/{len(target_urls)}] ⏭️ Đã có dữ liệu luật này, bỏ qua: {law['law_title']}")
                continue
            seen_titles.add(law['law_title'])
            
            all_data.append(law)
            valid_laws += 1
            total_articles += law['num_articles']
            print(f"[{i+1}/{len(target_urls)}] ✅ Xong: {law['law_title']} (Các điều: {law['num_articles']})")
            
            # Khúc này tự động lưu liên tục, lỡ bạn tắt ngang (Ctrl+C) thì không bị mất dữ liệu nãy giờ chạy
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
            print(f"[{i+1}/{len(target_urls)}] ❌ Bỏ qua (không có nội dung phù hợp): {link}")
            
        time.sleep(random.uniform(1.5, 3.5))

if __name__ == "__main__":
    main()