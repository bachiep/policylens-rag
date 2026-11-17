---
doc_id: OPS-SLA-02
title: Production Service Level Objectives, Error Budgets & On-Call Escalation
domain: engineering
language: en
effective_date: 2026-01-12
owner: Site Reliability Engineering (SRE)
---

# OPS-SLA-02: Production Service Level Objectives, Error Budgets & On-Call Escalation

## 1. Service Tier Availability & Latency SLOs
Document `OPS-SLA-02` establishes production reliability targets across service criticality tiers:

- **Tier-0 (Core Payment, Checkout & Auth APIs):** Must achieve **99.95%** monthly availability (maximum **21.9 minutes** of unplanned downtime per 30-day window) and **p95 API latency < 250 ms** under normal traffic.
- **Tier-1 (Internal Ops, Reporting & RAG Policy APIs):** Must achieve **99.9%** monthly availability and **p95 API latency < 500 ms** (excluding downstream third-party LLM streaming completion time).

## 2. Error Budget Freeze Policy
Error budgets quantify acceptable unreliability (`100% - SLO`). When a Tier-0 or Tier-1 service consumes more than **80%** (`> 80%`) of its rolling 30-day error budget, an automated **deployment freeze** takes effect for non-emergency feature releases.

During an error budget freeze, engineering capacity must prioritize reliability fixes, query optimization, circuit breakers, or rollback hardening until the 7-day rolling error rate drops back within healthy limits.

## 3. On-Call Paging & Escalation SLAs
Production alerts routed through PagerDuty/Opsgenie follow strict acknowledgment deadlines:

- **Primary On-Call Acknowledgment:** Must acknowledge P1 or P2 pages within **5 minutes** (`5 phút`).
- **Secondary Escalation:** Unacknowledged pages escalate automatically to the secondary on-call engineer after **10 minutes**.
- **Management Escalation:** If still unacknowledged at **20 minutes**, paging escalates to the Engineering Manager and SRE Lead.
