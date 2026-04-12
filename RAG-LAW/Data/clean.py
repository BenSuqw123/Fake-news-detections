import re
import json
import os

def clean_article_text(text):
    if not text:
        return ""
    
    # 1. Xóa sạch sành sanh MỌI ký tự xuống dòng (\n), \r, \t
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    
    # 2. Xử lý chấm lửng rác
    text = re.sub(r"\.{4,}", "...", text)

    # 3. Xóa các khoảng trắng thừa
    text = re.sub(r"[ ]+", " ", text)
            
    return text.strip()

def clean_dataset(data):
    cleaned = []
    seen_articles = set()

    for item in data:
        # Xử lý trường 'article': Giữ nguyên toàn bộ tiêu đề (Không tự ý cắt qua dấu chấm)
        article_raw = item.get("article", "").replace("\r", " ").replace("\n", " ")
        article_name = re.sub(r"\s+", " ", article_raw).strip()
        article_name = re.sub(r"\.{4,}", "...", article_name)

        unique_key = (item["law_title"], article_name)
        if unique_key in seen_articles:
            continue
        seen_articles.add(unique_key)

        title = item.get("law_title", "").strip()
        if not title:
            continue

        # Loại bỏ luật bị cào sai theo yêu cầu
        if "Nghị định 91/2026/NĐ-CP" in title or "hướng dẫn Luật Giáo dục đại học" in title:
            continue

        content = clean_article_text(item["content"])
        if not content:
            continue

        cleaned.append({
            "law_title": title,
            "article": article_name,
            "content": content,
            "url": item["url"]
        })

    return cleaned

if __name__ == "__main__":
    data_path = r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_full.json"
    output_path = r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_cleaned.json"
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        print(f"Đang xử lý {len(raw_data)} bản ghi từ '{data_path}' ...")
        cleaned_data = clean_dataset(raw_data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        print(f"Đã dọn dẹp và lưu {len(cleaned_data)} bản ghi sạch tại: {output_path}")