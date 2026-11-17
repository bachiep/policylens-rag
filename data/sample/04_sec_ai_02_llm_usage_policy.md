---
doc_id: SEC-AI-02
title: Generative AI & External LLM Usage Policy
domain: security
language: en
effective_date: 2026-03-01
owner: AI Governance & Security Committee
---

# SEC-AI-02: Generative AI & External LLM Usage Policy

## 1. Approved Enterprise LLM Gateways
Policy `SEC-AI-02` governs the use of Large Language Models (LLMs), embedding APIs, and coding assistants across engineering and operations teams. Engineers may only send internal context through approved enterprise API gateways covered by Zero Data Retention (**ZDR**) contractual agreements.

Pasting `CONFIDENTIAL` or `RESTRICTED` customer PII, proprietary financial data, or production credentials into public consumer AI web interfaces is strictly prohibited. Local self-hosted models (such as Ollama endpoints inside the corporate VPC) are permitted for offline testing and sensitive document prototyping.

## 2. Prompt Injection & Action Guardrails
All user prompts and retrieved text chunks from external or user-uploaded documents must be treated as **untrusted data**, never as privileged system instructions.

### 2.1. Deterministic Business Enforcement
Automated AI workflows or tool-calling agents must enforce policy limits deterministically in application code rather than relying solely on system prompts. Any automated financial action, refund, or credit adjustment exceeding **5,000,000 VND** (or **$200 USD**) must enter a `PENDING_HUMAN_APPROVAL` queue and cannot execute autonomously.

## 3. Grounded Citations & Hallucination Control
Internal policy and SOP assistants built on RAG pipelines must attach verifiable source citations (`doc_id` and `chunk_id`) to factual claims and return an explicit **abstention response** (`is_abstained=True`) when retrieval confidence falls below the calibrated relevance threshold.

*(Tóm tắt tiếng Việt — SEC-AI-02: Nghiêm cấm đưa dữ liệu khách hàng hoặc bí mật nội bộ lên chatbot công cộng; mọi hành động tài chính tự động vượt `5,000,000 VND` bắt buộc chuyển duyệt thủ công và hệ thống RAG phải từ chối trả lời khi độ tin cậy truy xuất dưới ngưỡng.)*
