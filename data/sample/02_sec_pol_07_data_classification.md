---
doc_id: SEC-POL-07
title: Data Classification & Encryption Standard
domain: security
language: en
effective_date: 2026-02-01
owner: Security Architecture Team
---

# SEC-POL-07: Data Classification & Encryption Standard

## 1. Four-Tier Data Classification Model
Under policy `SEC-POL-07`, all corporate datasets, documents, logs, and backups must be labeled into one of four sensitivity tiers before storage or transmission:

- **PUBLIC:** Approved external marketing materials, public API docs, and open-source sample corpora.
- **INTERNAL:** Internal engineering SOPs, architecture diagrams, and team handbooks not intended for public distribution.
- **CONFIDENTIAL:** Vendor contracts, financial forecasts, internal audit summaries, and employee HR records.
- **RESTRICTED:** Customer Personally Identifiable Information (PII), payment card tokens, authentication secrets, private keys, and production database snapshots.

## 2. Encryption Standards & KMS Key Rotation
All `CONFIDENTIAL` and `RESTRICTED` data stored in databases, object storage, or vector indexes must be encrypted at rest using **AES-256-GCM** backed by a managed Key Management Service (KMS).

### 2.1. Transport Layer & Key Lifecycle
- Data in transit across both external and internal service boundaries must enforce **TLS 1.3** (`HTTPS`). Legacy `TLS 1.2` endpoints require a documented 90-day exception signed by the CISO.
- Customer-managed and service KMS master encryption keys must rotate automatically every **90 days** without downtime.

## 3. Secrets Hygiene & Log Masking
Committing `.env` files, API keys, OAuth client secrets, or database credentials into any Git repository (public or private) is strictly prohibited under `SEC-POL-07`. All repositories must configure `.gitignore` rules and secret-scanning pre-commit checks.

### 3.1. Production Log Redaction Rules
Structured application logs must never record raw passwords, JWT bearer tokens, or full national IDs. Phone numbers logged for support diagnostics must be masked to retain only the **last 4 digits** (for example, `***-***-8421`), and email addresses must mask the local part beyond the first two characters.
