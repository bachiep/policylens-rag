---
doc_id: ENG-SOP-08
title: Zero-Downtime Database Schema Migration Guide
domain: engineering
language: en
effective_date: 2026-02-15
owner: Data Platform & Backend Engineering
---

# ENG-SOP-08: Zero-Downtime Database Schema Migration Guide

## 1. Expand-and-Contract Zero-Downtime Pattern
Under `ENG-SOP-08`, all relational database schema changes (PostgreSQL / MySQL) must be authored as forward-only versioned migrations managed by **Flyway** or **Alembic**. Manual DDL execution directly against production databases is strictly forbidden.

### 1.1. Prohibition on Same-Release Breaking DDL
Destructive or backward-incompatible operations — including `DROP COLUMN`, `DROP TABLE`, `RENAME COLUMN`, or changing column data types (`ALTER TYPE`) — are prohibited in the same release as application code updates. Engineers must follow the **Expand-and-Contract** pattern across at least two releases (`N` adds the new column/backfill and dual-writes; `N+1` removes the deprecated column after verifying zero remaining readers).

## 2. Lock Safety & Pre-Migration Guardrails
Long-held `ACCESS EXCLUSIVE` locks can pile up application queries and cause cascading API timeouts during peak traffic.

### 2.1. Mandatory Timeouts & Batching Limits
- Every migration transaction must explicitly configure **`lock_timeout = 3s`** and **`statement_timeout = 30s`** so stalled DDL fails fast instead of blocking live user traffic.
- Data backfill scripts updating existing rows must process chunks of **<= 5,000 rows per transaction** with brief sleep intervals between batches.
- High-risk index creation on large production tables must use `CREATE INDEX CONCURRENTLY` (outside a transaction block), and any destructive cleanup DDL requires a verified point-in-time database snapshot beforehand.
