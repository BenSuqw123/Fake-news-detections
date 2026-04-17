import json
import os
import re

def clean_text(text):
    if not text: return ""
    text = text.replace('\\"', '"').replace("\\", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    return text.strip()

def get_pure_article(article_text):
    match = re.search(r'(Điều\s+\d+)', article_text)
    return match.group(1) if match else "Văn bản"

def split_by_clause(text):
    parts = re.split(r'(^|\s)(?=\d+\.\s)', text)
    clauses = []
    current_clause = ""
    for part in parts:
        if not part: continue
        if re.match(r'^\s*$', part) and not current_clause: continue
        if re.match(r'^\d+\.\s', part):
            if current_clause:
                clauses.append(current_clause.strip())
            current_clause = part
        else:
            current_clause += part
            
    if current_clause:
        clauses.append(current_clause.strip())
    if not clauses:
        return [text.strip()]
    return [c for c in clauses if len(c) > 5]

def split_with_overlap(text, max_words=120, overlap=30):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += (max_words - overlap)
        if len(words) - start < 25:
            remaining = " ".join(words[start:])
            if remaining and remaining not in chunks[-1]:
                chunks[-1] = (chunks[-1] + " " + remaining).strip()
            break
            
    return chunks

def process_chunking(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            print(f"File không tồn tại: {input_path}")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_chunks = []

        for idx, item in enumerate(data):
            law_title = clean_text(item.get("law_title", ""))
            raw_article = clean_text(item.get("article", ""))
            content = clean_text(item.get("content", ""))
            url = item.get("url", "")
            
            short_article = get_pure_article(raw_article)
            if not content: continue

            clauses = split_by_clause(content)
            
            for i, clause in enumerate(clauses):
                sub_chunks = split_with_overlap(clause, max_words=120, overlap=30)

                for j, sub_text in enumerate(sub_chunks):
                    contextual_text = f"Văn bản: {law_title}. {short_article}. Nội dung: {sub_text}"

                    chunk_id = f"{idx}_{i+1}_{j+1}"
                    all_chunks.append({
                        "id": chunk_id,
                        "law_title": law_title,
                        "article": short_article,
                        "url": url,
                        "text": contextual_text
                    })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"Đã tạo {len(all_chunks)} chunks thông minh!")

    except Exception as e:
        import traceback
        print(f"Lỗi chi tiết: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    INPUT = r"D:\Fake-news-detections\RAG_LAW\Data\law_articles_cleaned.json"
    OUTPUT = r"D:\Fake-news-detections\RAG_LAW\Data\law_chunks.json"
    process_chunking(INPUT, OUTPUT)