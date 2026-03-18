import wikipediaapi
import json
import time
import os

OUTPUT_FILE = r'D:\Fake-news-detections\RAG\HISTORY\wikimedia_5000.jsonl'
LIMIT = 5000

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
wiki_en = wikipediaapi.Wikipedia(
    user_agent='ResearchProject/1.0 (ngan@example.com)',
    language='en'
)

seen_titles = set()
count = 0

def get_articles_recursive(category_page, max_level=2, current_level=0):
    global count
    if count >= LIMIT or current_level > max_level:
        return

    print(f"--- Đang quét: {category_page.title} (Level {current_level}) ---")
    
    try:
        members = category_page.categorymembers
    except Exception as e:
        print(f"Lỗi truy cập category: {e}")
        return

    for p in members.values():
        if count >= LIMIT: break
        
        if p.ns == wikipediaapi.Namespace.MAIN and p.title not in seen_titles:
            try:
                content = p.text[:4000]
                if len(content) < 300: continue
                
                data = {
                    "title_en": p.title,
                    "content_en": content,
                    "url": p.fullurl,
                    "source": "enwiki_recursive"
                }
                
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                seen_titles.add(p.title)
                count += 1
                print(f"   [+] Đã lưu: {count}/{LIMIT} - {p.title}")
                time.sleep(0.05) 
            except Exception as e:
                print(f"Lỗi bài {p.title}: {e}")
        
        elif p.ns == wikipediaapi.Namespace.CATEGORY:
            get_articles_recursive(p, max_level, current_level + 1)

def main():
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
            print("Đã xóa file cũ để ghi mới.")
        except:
            print("File cũ đang mở ở chương trình khác, sẽ ghi đè.")

    HISTORY_CATEGORIES = ["Category:Military history", "Category:Ancient civilizations"]

    for cat_name in HISTORY_CATEGORIES:
        if count >= LIMIT: break
        cat_page = wiki_en.page(cat_name)
        get_articles_recursive(cat_page)

if __name__ == "__main__":
    main()