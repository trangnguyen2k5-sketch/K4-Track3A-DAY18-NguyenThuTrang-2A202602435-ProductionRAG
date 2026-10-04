# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Thu Trang  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Bảng ánh xạ các khái niệm trong bài giảng với mã nguồn thực tế đã xây dựng trong dự án:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 nhóm các câu có cosine similarity cao lại với nhau thay vì cắt cứng theo số ký tự. Giúp giữ trọn vẹn ngữ cảnh của từng điều khoản chính sách." |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | "Tạo liên kết cha - con (Parent: 2048, Child: 256 tokens). Child dùng để so khớp vector chính xác, Parent cấp ngữ cảnh mở rộng cho LLM trả lời." |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | "Phân tích cấu trúc Markdown/Heading (#, ##, ###) để gom section và lưu trữ metadata tiêu đề tương ứng vào từng chunk." |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "Thuật toán RRF với k=60 kết hợp điểm xếp hạng lexical (BM25 tách từ tiếng Việt qua underthesea) và dense vector (Qdrant bge-m3), giúp cân bằng giữa tìm từ khóa chính xác và tìm theo ngữ nghĩa." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Dùng mô hình BAAI/bge-reranker-v2-m3 chấm điểm tương đồng trực tiếp giữa cặp (Query, Document), rút gọn từ top 20 candidate xuống top 3 chunk chất lượng nhất cho LLM." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "Đánh giá toàn diện 4 chỉ số: Faithfulness (độ trung thực), Answer Relevancy (bám sát câu hỏi), Context Precision (độ chuẩn xác ngữ cảnh), Context Recall (độ phủ thông tin)." |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | "Sinh tự động Summary, Câu hỏi giả định (HyQA), Context giải thích vị trí trích dẫn và Metadata để làm giàu chunk trước khi nhúng vector, giảm thiểu triệt để hiện tượng Retrieval failure." |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. `ModuleNotFoundError: No module named 'numpy'` khi chạy lệnh pytest ngoài terminal do môi trường terminal mặc định trỏ về Python hệ thống thay vì virtual environment `.venv`.
  2. `NotFoundError: models/gemini-1.5-flash is not found for API version v1beta` do phiên bản model cũ đã ngừng cung cấp dịch vụ trên endpoint API.
  3. `RateLimitError: 429 Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15` khi chạy RAGAS evaluation trên tài khoản miễn phí.

- **Nguyên nhân gốc rễ & Cách debug:**
  - *Môi trường thực thi:* Sử dụng trực tiếp đường dẫn `./.venv/bin/python` và `./.venv/bin/pytest` để đảm bảo thực thi đúng trên các gói thư viện đã cài đặt.
  - *Lỗi Model Deprecated:* Viết kịch bản kiểm tra danh sách model khả dụng thông qua `genai.list_models()` và chuyển sang model ổn định `gemini-3.1-flash-lite` cùng `models/gemini-embedding-001`.
  - *Lỗi Rate Limit 429:* Thêm bộ đệm giới hạn tần suất (`InMemoryRateLimiter` với `GEMINI_RPM=12`) và thiết lập cơ chế cache đĩa cục bộ (`enrich_cache.json`) để không phải gọi lại API lặp lại khi chạy lại pipeline.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Nắm vững hơn về cơ chế xác thực và vòng đời hỗ trợ model của các nhà cung cấp LLM API (Google / OpenAI).
  - Hiểu sâu về cách tích hợp `RunConfig` trong Ragas để điều tiết số lượng worker chạy song song (`max_workers=2`) nhằm thích ứng linh hoạt với hạn mức Free Tier.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý ảo Tra cứu Văn bản Pháp luật & Quy chế Doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với RecursiveCharacterTextSplitter (chunk size 500), embed bằng OpenAI text-embedding-3-small và tìm kiếm vector thuần trên ChromaDB.
- **Vấn đề / Bottlenecks đang gặp:** 
  - Thường xuyên bị cắt đứt giữa các điều khoản, văn bản nghị định khiến câu trả lời bị cụt ý.
  - Khó tìm kiếm các từ khóa đặc thù như số hiệu nghị định, ngày tháng ban hành.
  - Dễ gặp hallucination đối với các câu hỏi so sánh giữa quy định cũ và mới.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Structure-aware Chunking** kết hợp **Hierarchical Chunking** để phân tách rõ ràng theo Điều/Khoản/Mục của văn bản quy phạm pháp luật, bảo toàn ngữ cảnh cha-con.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense Qdrant)** kết hợp **RRF**. BM25 xử lý xuất sắc các số hiệu văn bản chính xác (VD: "Nghị định 13/2023"), còn Dense search tìm kiếm tốt theo câu hỏi ngữ nghĩa người dùng.
3. **Reranking:** Tích hợp **Cross-Encoder Reranker** (`bge-reranker-v2-m3` hoặc `flashrank`) để tái xếp hạng top 20 candidate xuống top 3-5 ngữ cảnh chính xác nhất trước khi gửi vào prompt sinh câu trả lời.
4. **Evaluation:** Áp dụng định kỳ bộ metric **RAGAS 4 metrics** (đặc biệt theo dõi `Faithfulness` > 0.95 và `Context Precision` > 0.85) để kiểm soát chất lượng qua từng chu kỳ cập nhật văn bản.
5. **Enrichment:** Sử dụng **Auto Metadata Extraction** để gắn metadata về số hiệu, trạng thái hiệu lực (Còn hiệu lực / Hết hiệu lực / Bị thay thế) và ngày ban hành vào từng chunk.

#### 3. Timeline triển khai
- **Tuần 1:** Tái cấu trúc pipeline tiền xử lý dữ liệu và đánh chỉ mục (Chunking M1 + Hybrid Search M2).
- **Tuần 2:** Tích hợp Reranker M3, hoàn thiện Module Enrichment M5 và chạy bộ đánh giá tự động RAGAS M4 để nghiệm thu chất lượng.
