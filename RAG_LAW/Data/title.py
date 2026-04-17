import json

# 1. Đọc dữ liệu từ file gốc của bạn (giả sử tên file là data.json)
try:
    with open(r'D:\Fake-news-detections\RAG-LAW\Data\law_articles_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Lấy danh sách law_title duy nhất
    # Dùng set() để loại bỏ trùng lặp và sorted() để sắp xếp theo bảng chữ cái
    unique_titles = sorted(list(set(item['law_title'] for item in data if 'law_title' in item)))

    # 3. Lưu kết quả vào file txt
    with open(r'D:\Fake-news-detections\RAG-LAW\Data\law_title.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("DANH SÁCH CÁC LOẠI LUẬT CÓ TRONG DỮ LIỆU\n")
        f_out.write("="*40 + "\n")
        for i, title in enumerate(unique_titles, 1):
            f_out.write(f"{i}. {title}\n")
            
    print(f"Đã trích xuất thành công {len(unique_titles)} loại luật vào file 'danh_sach_luat.txt'")

except FileNotFoundError:
    print("Không tìm thấy file data.json. Hãy kiểm tra lại tên file.")