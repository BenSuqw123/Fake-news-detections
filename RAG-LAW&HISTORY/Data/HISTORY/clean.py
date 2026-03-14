import json
import re
import os

def clean_content(data):
    content = data.get('content_en') or data.get('content', '')
    
    noise_sections = [
        r'==\s*References\s*==.*', 
        r'==\s*See also\s*==.*',
        r'References\n.*',
        r'Further reading\n.*'
    ]
    for pattern in noise_sections:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    content = re.sub(r'\[\d+\]', '', content)
    
    content = content.replace('\f', ' ')
    content = re.sub(r'\s+', ' ', content)
    
    return {
        "title_en": data.get('title_en') or data.get('title', '').strip(),
        "content_en": content.strip(),
        "url": data.get('url', ''),
        "source": data.get('source', '')
    }

def process_files(input_path, output_path):
    count = 0
    duplicate_count = 0
    seen_urls = set()
    
    print(f"Đang bắt đầu làm sạch và lọc trùng từ: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file nguồn tại {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                raw_data = json.loads(line)
                url = raw_data.get('url')

                if url in seen_urls:
                    duplicate_count += 1
                    continue
                
                cleaned_data = clean_content(raw_data)
                
                if cleaned_data['content_en']:
                    f_out.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")
                    seen_urls.add(url)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"✨ Đã xử lý: {count} bài | Đã bỏ qua: {duplicate_count} bài trùng...")
            
            except Exception as e:
                print(f"Lỗi xử lý: {e}")
    print(f"Hoàn thành!")
    print(f"📊 Tổng số bài sạch lưu lại: {count}")
    print(f"🚫 Tổng số bài trùng bị xóa: {duplicate_count}")
    print(f"💾 File lưu tại: {output_path}")

INPUT_FILE = r"D:\Fake-news-detections\RAG\HISTORY\wikimedia_5000.jsonl"
OUTPUT_FILE = r"D:\Fake-news-detections\RAG\HISTORY\wikimedia_5000_clean.jsonl"

process_files(INPUT_FILE, OUTPUT_FILE)

INPUT_FILE = r"D:\Fake-news-detections\RAG\HISTORY\world_history.jsonl"
OUTPUT_FILE = r"D:\Fake-news-detections\RAG\HISTORY\world_history_clean.jsonl"

process_files(INPUT_FILE, OUTPUT_FILE)