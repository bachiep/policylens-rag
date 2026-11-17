---
doc_id: ENG-SOP-03
title: Quy trình Git Flow, Pull Request & Triển khai Production (CI/CD Release SOP)
domain: engineering
language: vi
effective_date: 2026-01-25
owner: Platform & DevOps Team
---

# ENG-SOP-03: Quy trình Git Flow, Pull Request & Triển khai Production (CI/CD Release SOP)

## 1. Quy chuẩn Branch, Pull Request & Code Review
Quy trình `ENG-SOP-03` áp dụng cho toàn bộ kho mã nguồn dịch vụ backend, frontend và hệ thống AI nội bộ:

- Nhánh `main` được bật khóa bảo vệ (**branch protection**): cấm push trực tiếp, mọi thay đổi phải đi qua Pull Request từ nhánh tính năng (`feat/...`, `fix/...`, `chore/...`) gắn với Issue cụ thể.
- Mỗi Pull Request bắt buộc vượt qua pipeline CI (`linter + unit/integration tests`), đạt độ bao phủ kiểm thử (**code coverage**) tối thiểu **>= 80%** ở tầng domain logic, và có tối thiểu **1 peer approval** từ thành viên trong đội (**2 approvals** đối với các dịch vụ thanh toán `FIN` hoặc `PAY`).
- Thông điệp commit tuân theo chuẩn **Conventional Commits** (`feat(...)`, `fix(...)`, `test(...)`, `docs(...)`).

## 2. Chiến lược Triển khai Canary & Tự động Rollback
Các bản phát hành lên môi trường production không được chuyển toàn bộ lưu lượng ngay lập tức mà phải triển khai theo mô hình **Canary Release**:

### 2.1. Lộ trình Tăng tải & Giám sát Canary
- Định tuyến lưu lượng thử nghiệm theo 3 nấc: **10% -> 50% -> 100%** với tổng thời gian quan sát tối thiểu **30 phút** trước khi chốt phát hành.
- Theo dõi đồng thời tỷ lệ lỗi HTTP `5xx`, độ trễ `p95` và mức tiêu thụ bộ nhớ/CPU trên dashboard quan trắc.

### 2.2. Ngưỡng Tự động Quay lui (Auto-Rollback Trigger)
Hệ thống CD phải kích hoạt **auto-rollback** về phiên bản ổn định trước đó ngay lập tức nếu trong cửa sổ **5 phút** liên tiếp xảy ra một trong hai điều kiện: tỷ lệ lỗi **HTTP 5xx > 1.0%** hoặc độ trễ **p95 tăng vượt 2x** mức cơ sở trước khi triển khai.
