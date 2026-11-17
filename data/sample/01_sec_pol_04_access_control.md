---
doc_id: SEC-POL-04
title: Quy định Quản lý Truy cập & Xác thực Đa yếu tố (IAM & MFA Policy)
domain: security
language: vi
effective_date: 2026-01-15
owner: Information Security Team
---

# SEC-POL-04: Quy định Quản lý Truy cập & Xác thực Đa yếu tố (IAM & MFA Policy)

## 1. Phạm vi và Nguyên tắc Tối thiểu quyền (Least Privilege)
Tài liệu `SEC-POL-04` áp dụng cho toàn bộ nhân viên chính thức, thực tập sinh, nhà thầu và tài khoản dịch vụ truy cập hạ tầng nội bộ, kho mã nguồn GitHub, bảng điều khiển Cloud và hệ thống dữ liệu sản xuất.

Mọi yêu cầu cấp quyền mới phải tuân thủ nguyên tắc tối thiểu quyền (**Least Privilege**) và phân tách nhiệm vụ (**Separation of Duties**). Quyền truy cập chỉ được cấp theo vai trò (RBAC), tuyệt đối không chia sẻ tài khoản cá nhân hoặc dùng chung mật khẩu giữa các thành viên.

### 1.1. Rà soát quyền định kỳ (Access Review)
Quản lý bộ phận và nhóm Bảo mật Thông tin thực hiện rà soát toàn bộ quyền truy cập production theo chu kỳ hàng quý (**90 ngày/lần**). Các tài khoản không phát sinh đăng nhập trong **45 ngày** liên tiếp sẽ tự động chuyển sang trạng thái vô hiệu hóa tạm thời.

## 2. Quy định Xác thực Đa yếu tố (MFA) & Phiên đăng nhập
Xác thực đa yếu tố (**MFA**) là bắt buộc đối với 100% tài khoản khi truy cập VPN, Single Sign-On (SSO), GitHub Enterprise, Cloud Console và bảng điều khiển quản trị nội bộ.

### 2.1. Phương thức MFA được phê duyệt
Hệ thống chỉ chấp nhận ứng dụng sinh mã **TOTP** (Authenticator) hoặc khóa phần cứng **FIDO2/WebAuthn** (Passkey/YubiKey). Nghiêm cấm sử dụng **SMS OTP** hoặc gọi điện thoại làm yếu tố xác thực thứ hai đối với tài khoản đặc quyền (**PAM**) và quyền triển khai production.

### 2.2. Thời gian hết hạn phiên (Session Timeout) và Khóa tài khoản
- Phiên đăng nhập tại cổng quản trị và hạ tầng production tự động hết hạn sau **15 phút** không có thao tác (`idle timeout`).
- Phiên làm việc SSO thông thường có thời hạn tối đa **8 giờ** trước khi yêu cầu xác thực lại.
- Tài khoản sẽ bị khóa tạm thời trong **30 phút** nếu nhập sai mật khẩu liên tiếp **5 lần**.

## 3. Tài khoản Đặc quyền (PAM) & Tài khoản Khẩn cấp (Break-Glass)
Quyền truy cập đặc quyền vào cơ sở dữ liệu sản xuất hoặc cụm máy chủ không được gán thường trực (standing privilege). Kỹ sư phải gửi yêu cầu **Just-In-Time (JIT)** trên hệ thống PAM và được trưởng nhóm hoặc trực ca on-call phê duyệt với thời lượng tối đa **4 giờ** cho mỗi phiên.

Tài khoản khẩn cấp (**Break-Glass Account**) chỉ được mở niêm phong trong sự cố P1 khi hệ thống SSO gặp gián đoạn. Mọi lần kích hoạt Break-Glass bắt buộc phải hoàn tất kiểm tra nhật ký (**audit review**) bởi CISO hoặc Security Lead trong vòng **24 giờ** kể từ thời điểm sử dụng.
