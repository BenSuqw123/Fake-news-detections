import json
import re
import os

def clean_legal_content(data):
    content = data.get('content_en', '')
    
    content = re.sub(r'[_]{5,}', '', content)
    content = re.sub(r'[-]{5,}', '', content)
    content = re.sub(r'[*]{5,}', '', content)

    content = re.sub(r'\(ECF No\.\s*\d+\)', '', content)
    content = re.sub(r'PageID\s*\d+', '', content)
    content = re.sub(r'Case\s*[:#]\s*[\d\w-]+', '', content, flags=re.IGNORECASE)
    
    content = content.replace('§', 'Section').replace('¶', 'Paragraph')
    content = re.sub(r'\[\d+\]', '', content)
    
    content = content.replace('\f', ' ')
    content = re.sub(r'\r\n', '\n', content)
    content = re.sub(r'[ \t]+', ' ', content) 
    content = re.sub(r'\n\s*\n', '\n\n', content) 
    
    return {
        "title": data.get('title_en', '').strip(),
        "content": content.strip(),
        "date_filed": data.get('date_filed'),
        "url": data.get('url', ''),
        "source": data.get('source', '')
    }

def process_legal_data(input_path, output_path):
    seen_urls = set()
    count = 0
    duplicates = 0
    
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            
            try:
                raw_item = json.loads(line)
                url = raw_item.get('url')
                
                if url in seen_urls:
                    duplicates += 1
                    continue
                
                cleaned_item = clean_legal_content(raw_item)
                content = cleaned_item['content']
                
                if content and len(content) > 200 and not content.endswith((',', 'and', 'the')):
                    f_out.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
                    seen_urls.add(url)
                    count += 1
                
                if count % 100 == 0:
                    print(f"📥 Đã xử lý {count} bài...")
                    
            except Exception as e:
                print(f"⚠️ Lỗi: {e}")


INPUT = r"D:\Fake-news-detections\RAG\LAW\courtlistener_6000_data.jsonl" 
OUTPUT = r"D:\Fake-news-detections\RAG\LAW\courtlistener_6000_clean.jsonl"

process_legal_data(INPUT, OUTPUT)