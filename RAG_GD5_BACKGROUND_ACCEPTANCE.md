# RAG Giai Đoạn 5 — Background Processing Pipeline Acceptance Report

## Trạng thái: PASS

Toàn bộ hệ thống Background Processing Pipeline cho RAG (Giai đoạn 5) đã được thiết kế, tích hợp, chạy thực nghiệm và vượt qua tất cả kiểm thử nghiệm thu.

---

## 1. Mục tiêu và Kiến trúc Triển khai

### 1.1 State Machine
Hệ thống quản lý trạng thái tài liệu dựa trên finite state machine gồm 8 trạng thái chặt chẽ:
- `UPLOADED`: Tài liệu PDF đã lưu trữ thành công trên đĩa.
- `QUEUED`: Đã đưa vào hàng đợi xử lý nền.
- `EXTRACTING`: Đang thực hiện trích xuất văn bản (Text Extraction via PyPDF/PdfReader).
- `EMBEDDING`: Đang thực hiện chunking và suy luận embedding (Jina Embeddings v2 trên CPU).
- `INDEXING`: Đang nạp vectors và metadata vào Qdrant Local.
- `COMPLETED`: Toàn bộ pipeline hoàn tất thành công, vector đã sẵn sàng cho truy vấn.
- `FAILED`: Thất bại ở một công đoạn cụ thể, ghi rõ `failure_stage` và `error_message`.
- `CANCELLED`: Tác vụ đã được hủy theo yêu cầu người dùng hoặc shutdown.

### 1.2 Quản lý Tiến trình và Concurrency Control
- **Single-Worker Concurrency (`concurrency = 1`)**: Do Qdrant Local Persistent Mode sử dụng cơ chế file lock (`portalocker`), việc nhiều worker ghi đồng thời sẽ gây xung đột khóa. Hàng đợi xử lý được điều phối tuần tự qua `asyncio.Queue` kết hợp khóa re-entrant `threading.RLock()`.
- **Non-blocking Event Loop**: Các tác vụ tính toán nặng hoặc đồng bộ (như `extract_document` và `index_document`) được đẩy vào thread pool qua `asyncio.to_thread()`, bảo đảm API server và health endpoint không bao giờ bị nghẽn (non-blocking).
- **Atomic Persistence**: Trạng thái tác vụ được lưu trữ bền vững tại `{document_id}.processing.json` thông qua cơ chế ghi file tạm rồi replace nguyên tử (`.processing.tmp` $\to$ `.processing.json`), ngăn ngừa hỏng file khi mất điện đột ngột hoặc dừng server.
- **Crash Recovery**: Khi ứng dụng khởi động lại, `processing_service.recover_interrupted_tasks()` tự động quét thư mục upload, phát hiện các tác vụ bị gián đoạn giữa chừng (`QUEUED`, `EXTRACTING`, `EMBEDDING`, `INDEXING`) và chuyển sang `FAILED` kèm thông báo cho phép retry an toàn.

---

## 2. API Endpoints Giai đoạn 5

| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/rag/documents/upload?auto_process=true` | Upload file PDF, hỗ trợ cờ `auto_process` (mặc định `false` để giữ tương thích). |
| `GET` | `/rag/documents` | Liệt kê danh sách tất cả tài liệu, trạng thái hiện tại, kích thước và số chunk đã lập chỉ mục. |
| `POST` | `/rag/documents/{document_id}/process` | Đưa tài liệu vào hàng đợi xử lý nền (hỗ trợ `force_reprocess: bool`). |
| `GET` | `/rag/documents/{document_id}/status` | Truy vấn trạng thái chi tiết, tiến độ, thời gian bắt đầu/kết thúc, lỗi nếu có. |
| `POST` | `/rag/documents/{document_id}/cancel` | Yêu cầu hủy tác vụ đang chờ trong hàng đợi hoặc đang xử lý. |

---

## 3. Bằng chứng Thực nghiệm Nghiệm thu (Real Evidence)

Dữ liệu ghi nhận từ kịch bản chạy thật không dùng mock (`scripts/run_rag_full_acceptance.py`):

1. **Scenario A — Upload kèm Auto-Processing**:
   - File: `tinyllama_rag_guide.pdf` (2 trang, tài liệu tiếng Anh).
   - Tiến trình: `UPLOADED` $\to$ `QUEUED` $\to$ `EXTRACTING` $\to$ `EMBEDDING` $\to$ `INDEXING` $\to$ `COMPLETED`.
   - Kết quả: **2 chunks** float32 (768 chiều) lập chỉ mục thành công vào collection `knowledgebase`.
   - Thời gian xử lý: **6.66 giây**.

2. **Scenario B — Upload thủ công và Enqueue**:
   - File: `fastapi_qdrant_vn.pdf` (2 trang, tài liệu tiếng Việt).
   - Trạng thái ban đầu: `UPLOADED`.
   - Gọi `POST /rag/documents/{id}/process`: Chuyển sang `QUEUED` và hoàn tất `COMPLETED`.
   - Kết quả: **2 chunks** lập chỉ mục thành công.
   - Thời gian xử lý: **2.02 giây**.

3. **Scenario C — Ngăn chặn trùng lặp và Xung đột (Conflict Prevention)**:
   - Khi tài liệu đang trong queue/active hoặc đã `COMPLETED` (mà không có `force_reprocess=True`), hệ thống trả về HTTP 409 hoặc trạng thái hiện tại mà không tạo task trùng lặp.

4. **Scenario D — Hủy tác vụ và Phục hồi sự cố (Crash Recovery)**:
   - Đã kiểm tra chức năng hủy `cancel_processing()`.
   - Đã mô phỏng tác vụ bị ngắt quãng do tắt server: hàm phục hồi `recover_interrupted_tasks()` đã nhận diện và phục hồi trạng thái chính xác.

---

## 4. Kiểm thử Tự động (Automated Tests)

Test suite `tests/test_rag_processing.py` đạt **7/7 passed**:
- `test_get_status_non_existent`: 404 cho UUID không tồn tại.
- `test_get_status_uploaded_file`: Trả về trạng thái `UPLOADED` cho file PDF có sẵn.
- `test_enqueue_processing_success`: Trạng thái `QUEUED` và xung đột HTTP 409 khi enqueue lần 2.
- `test_cancel_processing`: Hủy tác vụ thành công, trạng thái `CANCELLED`.
- `test_list_documents`: Danh sách tài liệu phản ánh chính xác metadata.
- `test_upload_with_auto_process`: Cờ `auto_process=true` tự động kích hoạt hàng đợi xử lý.
- `test_recover_interrupted_tasks`: Phục hồi tác vụ bị gián đoạn khi restart server.
