import re
import json

def clean_article_text(text):
    if not text:
        return ""
    
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"([a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9,])\n([a-zđáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])", r"\1 \2", text)

    text = re.sub(r"(?<!Điều\s)(?<!Khoản\s)(?<!Điểm\s)(\b\d+\.)\s", r"\n\1 ", text)
    
    text = re.sub(r"\n+", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    text = " ".join(unique_lines)
    
    return text

def clean_dataset(data):
    cleaned = []

    for item in data:
        content = clean_article_text(item["content"])

        cleaned.append({
            "law_title": item["law_title"].strip(),
            "article": item["article"].strip(),
            "content": content,
            "url": item["url"]
        })

    return cleaned

if __name__ == "__main__":
    data=r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_full.json"
    output=r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_cleaned.json"
    try:
        with open(data, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        print(f"Xử lý {len(raw_data)} bản ghi...")
        cleaned_data = clean_dataset(raw_data)
        
        with open(output, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
        print(f"Dữ liệu sạch được lưu tại: law_articles_cleaned.json")
    except FileNotFoundError:
        print("Không tìm thấy file law_articles_full.json để dọn dẹp. Lời khuyên: Hãy đảm bảo bạn đã chạy file crawl.py trước để có dữ liệu.")

