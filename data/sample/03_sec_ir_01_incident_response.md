---
doc_id: SEC-IR-01
title: Quy trình Ứng phó Sự cố An ninh mạng (Incident Response Playbook)
domain: security
language: vi
effective_date: 2026-01-20
owner: CSIRT / Blue Team
---

# SEC-IR-01: Quy trình Ứng phó Sự cố An ninh mạng (Incident Response Playbook)

## 1. Phân loại Mức độ Sự cố (Severity Levels)
Tài liệu `SEC-IR-01` quy định chuẩn phân loại và điều phối xử lý sự cố bảo mật cho toàn bộ hạ tầng sản phẩm:

- **P1 (Critical):** Nghi ngờ hoặc xác nhận rò rỉ dữ liệu `RESTRICTED` của khách hàng, lộ khóa bí mật production, mã độc tống tiền (ransomware), hoặc gián đoạn diện rộng dịch vụ lõi do tấn công mạng.
- **P2 (High):** Phát hiện lỗ hổng nghiêm trọng đang bị thăm dò, xâm nhập giới hạn trên một dịch vụ nội bộ không chứa dữ liệu khách hàng, hoặc tấn công từ chối dịch vụ mức trung bình.
- **P3 (Medium):** Cảnh báo bất thường đơn lẻ, email lừa đảo (phishing) chưa có dấu hiệu lộ thông tin xác thực.

## 2. Thời hạn Phản hồi & Cô lập (Response SLAs)
Tốc độ phản ứng quyết định mức độ thiệt hại trong 1 giờ đầu tiên của sự cố:

### 2.1. Thông báo CISO và Cô lập Hạ tầng
- Với sự cố **P1**, kỹ sư trực bảo mật phải xác nhận sự cố và thông báo trực tiếp tới **CISO** cùng Incident Commander trong vòng **15 phút**.
- Thời gian cô lập sự cố (**Containment SLA** — thu hồi token bị lộ, ngắt kết nối máy chủ nhiễm bẩn, chặn IP tấn công) tối đa là **30 phút** đối với P1 và **2 giờ** đối với P2.

### 2.2. Bảo toàn Bằng chứng Pháp lý Số (Forensics Preservation)
Tuyệt đối không xóa máy ảo hoặc khởi tạo lại container bị xâm nhập trước khi thu thập ảnh đĩa và bộ nhớ (**forensic snapshot**). Bản sao bằng chứng và nhật ký hệ thống phải được lưu giữ ở vùng lưu trữ chỉ đọc (**WORM storage**) tối thiểu **72 giờ** (và lưu trữ dài hạn 1 năm với sự cố P1).

## 3. Khôi phục & Báo cáo Sau Sự cố (Post-Mortem)
Hệ thống chỉ được phép đưa trở lại phục vụ người dùng sau khi vá triệt để lỗ hổng gốc, xoay vòng toàn bộ khóa bí mật liên quan và vượt qua kiểm tra xác minh của đội CSIRT.

Trong vòng **5 ngày làm việc** kể từ khi đóng sự cố P1 hoặc P2, nhóm chịu trách nhiệm bắt buộc hoàn thành tài liệu phân tích nguyên nhân gốc không đổ lỗi (**Blameless Post-Mortem / RCA**) kèm danh sách hành động phòng ngừa có người phụ trách và hạn hoàn thành cụ thể.
