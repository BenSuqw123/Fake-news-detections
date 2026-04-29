import re
import json
import os

def fix_vietnamese_broken_words(text):
    pattern = r'\b([bcdfghjklmnpqrstvwxzđBCDFGHJKLMNPQRSTVWXZĐ]{1,3})\s+([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]+)\b'
    return re.sub(pattern, r'\1\2', text)

def clean_text_perfectly(text):
    if not text:
        return ""
    text = text.replace("\\", "")
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = fix_vietnamese_broken_words(text)
    text = re.sub(r'\.{4,}', '...', text)
    return text.strip()

def clean_dataset(data):
    cleaned = []
    seen_keys = set()
    stats = {}

    for item in data:
        title = item.get("law_title", "").strip()
        url = item.get("url", "")
        
        if not title or not item.get("content"):
            continue

        if any(bad_word in title for bad_word in ["Nghị định 91/2026/NĐ-CP", "hướng dẫn Luật Giáo dục đại học"]):
            continue

        article_name = clean_text_perfectly(item.get("article", ""))
        content = clean_text_perfectly(item.get("content", ""))

        unique_key = (title, article_name)
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        cleaned.append({
            "law_title": title,
            "article": article_name,
            "content": content,
            "url": url
        })
        
        stats[title] = stats.get(title, 0) + 1

    return cleaned, stats

if __name__ == "__main__":
    data_path = r"C:\Users\User\Desktop\Fake news detection\Fake-news-detections\RAG_LAW\Data\law_articles.json"
    output_path = r"C:\Users\User\Desktop\Fake news detection\Fake-news-detections\RAG_LAW\Data\law_articles_cleaned.json"
    report_path = r"C:\Users\User\Desktop\Fake news detection\Fake-news-detections\RAG_LAW\Data\law_articles_cleaned_summary.txt"

    if os.path.exists(data_path):
        print("Đang đọc dữ liệu gốc...")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"Đang xử lý dọn dẹp và vá lỗi chính tả cho {len(raw_data)} bản ghi...")
        cleaned_data, stats = clean_dataset(raw_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== BÁO CÁO DỮ LIỆU LUẬT (CLEANED) ===\n\n")
            for name, count in stats.items():
                f.write(f"- {name}: {count} điều\n")
            f.write(f"\n=> TỔNG CỘNG: {len(cleaned_data)} điều luật hợp lệ.\n")

        print(f"\n[THÀNH CÔNG] Dữ liệu RAG đã sẵn sàng!")
        print(f"- File JSON: {output_path}")
        print(f"- Báo cáo: {report_path}")
    else:
        print(f"LỖI: Không tìm thấy file {data_path}")