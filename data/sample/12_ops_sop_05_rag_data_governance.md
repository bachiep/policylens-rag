---
doc_id: OPS-SOP-05
title: Quy chuẩn Vận hành Kho Tri thức & Đánh giá RAG (Knowledge Base & RAG Governance)
domain: engineering
language: vi
effective_date: 2026-03-05
owner: Applied AI Engineering Team
---

# OPS-SOP-05: Quy chuẩn Vận hành Kho Tri thức & Đánh giá RAG (Knowledge Base & RAG Governance)

## 1. Chuẩn hóa Dữ liệu & Chiến lược Cắt đoạn (Chunking)
Tài liệu `OPS-SOP-05` chuẩn hóa toàn bộ quy trình nạp dữ liệu và kiểm soát chất lượng cho các trợ lý tra cứu tài liệu nội bộ (RAG services):

- **Chuẩn hóa văn bản:** Mọi tài liệu tiếng Việt và tiếng Anh bắt buộc chuẩn hóa Unicode về dạng **NFC** (`unicodedata.normalize("NFC")`) trước khi tách từ, tính mã băm hoặc tạo embedding nhằm tránh lỗi lệch dấu thanh điệu giữa các hệ điều hành.
- **Chiến lược Chunking:** Sử dụng bộ cắt đoạn nhận biết cấu trúc tiêu đề Markdown (**Markdown heading-aware recursive chunker**) với kích thước tiêu chuẩn **800–1,000 ký tự** (~200–250 tokens) và phần chồng lấn (**chunk_overlap**) khoảng **15%** (**150 ký tự**). Mỗi chunk bắt buộc lưu kèm đường dẫn tiêu đề cha (`section_path`) và vị trí ký tự (`char_start`, `char_end`) để phục vụ kiểm chứng trích dẫn gốc.

## 2. Chống nạp trùng & Đánh giá Định lượng (Evals)
Quy trình nạp dữ liệu phải đảm bảo tính lũy đẳng (**idempotent ingestion**): mỗi tài liệu được băm nội dung bằng thuật toán **SHA-256** (`doc_hash`); các lần chạy lại sẽ bỏ qua tài liệu không đổi và chỉ cập nhật các chunk của tài liệu có nội dung mới.

### 2.1. Ngưỡng Đạt chuẩn trước khi Phát hành (Release Gates)
Mọi thay đổi về chiến lược cắt đoạn, công thức hợp nhất **Hybrid RRF** (`dense + BM25`), mô hình **Reranker** hoặc ngưỡng từ chối trả lời (**Abstention Threshold**) phải chạy lại bộ kiểm thử định lượng (`evals/dataset.jsonl`) và đáp ứng tối thiểu:
- **HitRate@5 >= 0.85** trên tập câu hỏi nội bộ song ngữ.
- **Abstention Accuracy >= 0.90** đối với nhóm câu hỏi bẫy ngoài phạm vi tài liệu (không tự bịa đặt chính sách).
- Không rò rỉ dữ liệu test có sử dụng API trả phí trong đường ống CI tự động.
