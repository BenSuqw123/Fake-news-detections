import wikipediaapi
import json
import time
import os

OUTPUT_FILE = r'C:\Users\KimNgan\Desktop\Fake news\data_ground\history\wikimedia_en_raw_2000.jsonl'
LIMIT = 5000

wiki_en = wikipediaapi.Wikipedia(
    user_agent='FakeNewsResearchProject/1.0 (your-email@example.com)',
    language='en'
)

seen_titles = set()
count = 0

def get_articles_recursive(category_page, max_level=2, current_level=0):
    global count
    
    if count >= LIMIT or current_level > max_level:
        return

    print(f"\n>>> Level {current_level} - Scanning: {category_page.title}")
    
    members = category_page.categorymembers
    
    for p in members.values():
        if count >= LIMIT: break
        
        if p.ns == wikipediaapi.Namespace.MAIN and p.title not in seen_titles:
            try:
                content = p.summary if len(p.summary) > 500 else p.text[:4000]
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
                if count % 10 == 0:
                    print(f"   [+] Progress: {count}/{LIMIT} - Saved: {p.title}")
                
                time.sleep(0.05) 
                
            except Exception as e:
                print(f"Error at {p.title}: {e}")
        
        elif p.ns == wikipediaapi.Namespace.CATEGORY:
            get_articles_recursive(p, max_level, current_level + 1)

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    HISTORY_CATEGORIES = [
        "Category:Ancient civilizations",
        "Category:Military history",
        "Category:History by period",
        "Category:Historical events",
        "Category:Biographies"
    ]

    print(f"--- BẮT ĐẦU CÀO ĐỆ QUY {LIMIT} BÀI ---")
    
    for cat_name in HISTORY_CATEGORIES:
        if count >= LIMIT: break
        cat_page = wiki_en.page(cat_name)
        if cat_page.exists():
            get_articles_recursive(cat_page)

    print(f"\n--- HOÀN THÀNH! Tổng cộng lấy được: {count} bài ---")

if __name__ == "__main__":
    main()