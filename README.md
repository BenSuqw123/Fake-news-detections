# Vietnamese Fake News Detection System

Hệ thống kiểm chứng thông tin pháp luật tiếng Việt sử dụng công nghệ **Hybrid RAG** (Retrieval-Augmented Generation) kết hợp giữa tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa (ChromaDB), được tối ưu hóa cho Llama 3.2.

---

## PHẦN 1 — ETL PIPELINE (Dữ Liệu Nền Tảng)

### 1.1 Ingestion & Cleaning
- **Nguồn dữ liệu:** Crawl tự động từ `thuvienphapluat.vn` với 11 văn bản luật 
- **Chuẩn hóa:** Xử lý Encoding UTF-8, loại bỏ các ký tự đặc biệt và nhiễu HTML để đảm bảo chất lượng embedding.

### 1.2 Chunking Chiến Thuật
- **Sliding Window:** Sử dụng cửa sổ trượt (window size: 120 words, overlap: 20) để không làm mất ngữ cảnh giữa các đoạn luật.
- **Metadata Mapping:** Mỗi chunk được gắn kèm ID điều luật, tên văn bản và URL gốc để phục vụ trích dẫn bằng chứng.

### 1.3 Dual Indexing (Chỉ mục kép)
- **BM25 (Lexical):** Sử dụng tokenizer chuyên dụng cho tiếng Việt (`underthesea`) để bắt chính xác các thuật ngữ pháp lý đặc thù.
- **ChromaDB (Vector):** Sử dụng model `bge-m3` để hiểu các truy vấn diễn đạt theo ý hiểu của người dùng mà không cần trùng khớp từ khóa.

---

## PHẦN 2 — RAG PIPELINE (Xử lý Truy vấn)

### Bước 1 — Guardrail & Domain Filtering
- **Model:** `llama3.2` (Temperature = 0).
- **Nhiệm vụ:** Kiểm tra xem câu hỏi có thuộc phạm vi Pháp luật/Lịch sử hay không. Tự động từ chối (`REJECT`) các nội dung mang tính xúc phạm, chính trị nhạy cảm hoặc nằm ngoài phạm vi 11 văn bản luật hỗ trợ.

### Bước 2 — Hybrid Retrieval & RRF Fusion
- **Support Retrieval**: Tìm kiếm Top-10 tài liệu có khả năng ủng hộ tuyên bố nhất.
- **Contradiction Retrieval**: Tự động tạo truy vấn đối nghịch (thêm các từ khóa "cấm", "vi phạm", "không được") để tìm kiếm Top-5 bằng chứng phản biện.
- **RRF (Reciprocal Rank Fusion)**: Kết hợp kết quả từ BM25 và Vector Store theo trọng số để đưa ra danh sách bằng chứng tối ưu nhất cho LLM.

### Bước 3 — LLM Reasoning Evaluation
LLM thực hiện đánh giá sâu dựa trên 3 tiêu chí:
1. **Context Matching**: Đối chiếu xem đối tượng trong luật và đối tượng trong tuyên bố có khớp nhau không.
2. **Chain of Reasoning**: Xây dựng lập luận logic nếu tuyên bố cần kết hợp thông tin từ nhiều điều luật.
3. **Evidence Validation**: Trích xuất chính xác số hiệu Điều/Khoản làm bằng chứng trực tiếp.

### Bước 4 — Verdict Aggregation (Tổng hợp kết quả)
Hệ thống sử dụng bộ quy tắc trọng số (`verdict_aggregator.py`) để đưa ra kết luận cuối cùng:
- **SUPPORTED**: Tìm thấy bằng chứng trực tiếp và LLM xác nhận đúng (Confidence > 80%).
- **CONTRADICTED**: Tìm thấy bằng chứng mâu thuẫn trực tiếp với tuyên bố.
- **PARTIAL**: Tuyên bố có phần đúng nhưng thiếu điều kiện hoặc có phần sai đi kèm.
- **INSUFFICIENT**: Hệ thống không tìm thấy đủ dữ liệu trong 11 văn bản luật để đưa ra kết luận.
- **ERROR**: input bị guardrail từ chối hoặc pipeline gặp lỗi.
---

## PHẦN 3 — CÀI ĐẶT & KHỞI CHẠY

### 1. Yêu cầu hệ thống
- **Ollama**: Đã cài đặt và chạy server local.
- **Models**:
  ```bash
  ollama pull llama3.2
  ollama pull bge-m3
  ```
- **Python**: Phiên bản 3.11 trở lên.

### 2. Cài đặt môi trường
```bash
pip install -r requirements.txt
```
tạo file .env dùng qroq tạo API_KEY
GROQ_API_KEY=your_key_here

### 3. Chạy Web API (FastAPI)
```powershell
uvicorn api.main:app --reload
```
Hệ thống sẽ cung cấp UI tại: `http://localhost:8000`

---

