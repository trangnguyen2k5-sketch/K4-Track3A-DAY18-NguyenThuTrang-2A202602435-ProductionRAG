# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Thu Trang  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---|---|---|
| Faithfulness | 0.9400 | 0.9938 | +0.0538 |
| Answer Relevancy | 0.7655 | 0.7818 | +0.0163 |
| Context Precision | 0.7000 | 0.8083 | +0.1083 |
| Context Recall | 0.8250 | 0.8417 | +0.0167 |

> Nhận xét: Pipeline Production cải thiện rõ rệt nhất ở `Context Precision` (+10.83%) và `Faithfulness` đạt gần như tuyệt đối (99.38%), chứng minh vai trò hiệu quả của Hybrid Search kết hợp Reranker và Chunk Enrichment.

---

## Bottom-5 Failures

Dưới đây là phân tích chi tiết cho 5 câu hỏi có điểm số thấp nhất từ `reports/ragas_report.json`:

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Cần Giám đốc bộ phận phê duyệt và cần xác nhận cấu hình kỹ thuật từ phòng CNTT (thiếu chi tiết về 3 báo giá).
- **Worst metric:** `context_precision` (Score: 0.0, Avg Score: 0.5236)
- **Error Tree:** Output đúng một phần → Context đúng một phần (thiếu chunk về quy định 3 báo giá) → Query đa ý phức tạp.
- **4 câu hỏi phân tích:**
  1. *Câu trả lời của mô hình có đúng không?* Trả lời đúng thẩm quyền phê duyệt và yêu cầu cấu hình CNTT, nhưng thiếu sót điều kiện "cần ít nhất 3 báo giá đối với đơn hàng trên 10 triệu".
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Các chunk retrieve về chứa tài liệu phê duyệt chi tiêu và cấp phát laptop, nhưng chunk chứa quy tắc mua sắm chung (quy định 3 báo giá) bị rơi xuống dưới top-k.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có thể áp dụng Sub-query Decomposition / Multi-query vì câu hỏi ghép nhiều vế: vừa hỏi cấp duyệt, vừa hỏi điều kiện phòng CNTT, vừa liên quan mức giá 30 triệu.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 2 (Search Retrieval) và Module 3 (Reranker) - cần bổ sung Query Expansion / Multi-query Retrieval để bao phủ cả quy trình mua sắm lẫn thiết bị CNTT.

---

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** 18 ngày phép năm; lương Senior từ 20 đến 35 triệu VNĐ.
- **Worst metric:** `context_precision` (Score: 0.0, Avg Score: 0.5369)
- **Error Tree:** Output đúng hoàn toàn → Context retrieved thừa nhiều chunk rác không liên quan giữa 2 văn bản khác nhau (chính sách nghỉ phép vs khung lương).
- **4 câu hỏi phân tích:**
  1. *Câu trả lời của mô hình có đúng không?* Mô hình trả lời hoàn toàn chính xác cả 2 vế số ngày phép (18 ngày) và khung lương (20-35 triệu).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đầy đủ cả 2 thông tin, nhưng các chunk liên quan nằm rải rác và bị xen kẽ bởi nhiều chunk không liên quan khác từ top-20 search khiến `context_precision` bị phạt về 0.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Đây là câu hỏi dạng multi-hop / multi-domain nối giữa HR chính sách phép và Finance/HR bảng lương. Tách thành 2 câu hỏi con sẽ giúp retrieval sạch hơn nhiều.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 3 (Reranker) - cần tinh chỉnh Reranker để loại bỏ các chunk trung gian không liên quan hoặc tách sub-queries trước khi retrieve.

---

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** 17.000.000 VNĐ (Junior max 20 triệu * 85%).
- **Worst metric:** `context_precision` (Score: 0.3333, Avg Score: 0.7719)
- **Error Tree:** Output đúng → Context đúng → Nhưng thứ hạng chunk chứa quy định 85% thử việc không nằm ở top 1.
- **4 câu hỏi phân tích:**
  1. *Câu trả lời của mô hình có đúng không?* Mô hình trả lời đúng 17 triệu và tính toán chuẩn xác.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đáp án (bảng lương Junior và điều khoản 85% lương thử việc), tuy nhiên chunk chứa bảng lương Junior xuất hiện sau chunk nói chung về chính sách tuyển dụng.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rõ ràng, không cần viết lại.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 1 (Chunking) & Module 2 (BM25 Tokenization) - Cần đảm bảo các bảng dữ liệu (Table) không bị phân mảnh rời rạc giữa tiêu đề cột và nội dung lương.

---

### #4
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Dữ liệu Bí mật (Cấp độ 3).
- **Worst metric:** `context_recall` (Score: 0.5, Avg Score: 0.7788)
- **Error Tree:** Output đúng cốt lõi → Context thiếu một phần tài liệu (thiếu tài liệu Quy chế chi trả lương, chỉ lấy được tài liệu An toàn thông tin).
- **4 câu hỏi phân tích:**
  1. *Câu trả lời của mô hình có đúng không?* Trả lời đúng cấp độ Bí mật (Cấp 3).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Chỉ trích xuất được chunk từ văn bản Phân loại an toàn thông tin, thiếu mất chunk từ Quy chế chi trả lương (nơi nhấn mạnh việc cấm tiết lộ với đồng nghiệp).
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi có thể bổ sung ngữ cảnh: "Theo quy chế lương và quy định an toàn thông tin, thông tin lương thuộc cấp độ nào?".
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 2 (Search) - Cần cải thiện Hybrid Search (tăng trọng số BM25 cho cụm từ khóa chuyên biệt "thông tin lương", "quy chế chi trả lương").

---

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, bắt buộc kích hoạt MFA cho email, VPN và các hệ thống nội bộ theo quy định v2.0.
- **Worst metric:** `context_recall` (Score: 0.5, Avg Score: 0.8132)
- **Error Tree:** Output đúng chính sách mới → Context thiếu đối chiếu với chính sách cũ v1.0.
- **4 câu hỏi phân tích:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng kết luận thực tế hiện hành (Bắt buộc kích hoạt MFA).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Retrieval lấy được chunk của chính sách v2.0, nhưng bỏ sót văn bản cũ v1.0 (trong ground truth có đề cập so sánh giữa v2.0 và v1.0).
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi ngắn gọn, thực tế người dùng hỏi như vậy là bình thường.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 5 (Enrichment - Auto Metadata) & Module 2 - Cần đánh dấu metadata `version: "v2.0"` và `status: "active"` để ưu tiên truy xuất và phân biệt với phiên bản cũ.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → CÓ. Mô hình trả lời đúng 18 ngày phép và khung lương 20 - 35 triệu đồng/tháng.
2. **Context đúng?** → CÓ nhưng BỊ LOÃNG. Chứa cả 2 chunk cần thiết nhưng xen kẽ 3 chunk rác về bảo hiểm và quy chế thưởng, khiến `context_precision` bị điểm thấp (0.0).
3. **Query rewrite OK?** → Câu hỏi ghép 2 thực thể không cùng văn bản nguồn: "nghỉ phép" (HR Policy) và "khung lương Senior" (Salary Scale).
4. **Fix ở bước:**  
   - Bổ sung bước **Query Decomposition** (tách thành: Query A: "Chính sách thâm niên ngày phép năm nhân viên" và Query B: "Khung lương bậc Senior").
   - Sau đó thực hiện truy vấn song song và merge contexts lại trước khi rerank.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp kỹ thuật **Sub-Query Decomposition** để tự động bóc tách các câu hỏi phức hợp thành các sub-queries đơn giản.
- Thêm **Document Version Filtering** thông qua metadata filtering ở Module 2 để lọc bỏ hoàn toàn các chính sách cũ đã hết hiệu lực (v2023 vs v2024).
