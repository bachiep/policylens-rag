---
doc_id: HR-POL-05
title: Employee Onboarding, Laptop Provisioning & Offboarding Checklist
domain: hr
language: en
effective_date: 2026-01-10
owner: People Ops & IT Support
---

# HR-POL-05: Employee Onboarding, Laptop Provisioning & Offboarding Checklist

## 1. Pre-Boarding & Laptop Provisioning
Policy `HR-POL-05` defines IT asset readiness and lifecycle controls for all joiners and leavers. Hiring managers must submit the standardized IT provisioning request at least **T-5 business days** prior to the candidate's confirmed start date.

### 1.1. Workstation Security Baseline
Every corporate engineering laptop is provisioned with Mobile Device Management (MDM), active Endpoint Detection and Response (EDR), and mandatory full-disk encryption (**FileVault** on macOS or **BitLocker** on Windows). Installing unauthorized operating systems or disabling disk encryption is prohibited.

## 2. Day-1 Account Activation
New hires receive corporate email, SSO credentials, and baseline repository access on Day 1 only after completing the mandatory 30-minute security orientation and configuring hardware or TOTP multi-factor authentication per `SEC-POL-04`.

## 3. Offboarding & Access Revocation SLA
Protecting corporate repositories and customer data during employee departures requires strict revocation timelines:

- **Voluntary Resignation:** IT Security must revoke SSO sessions, VPN certificates, corporate email, and GitHub/Cloud API tokens within **2 hours** of the official termination timestamp.
- **Involuntary Termination:** Access revocation must execute immediately within **15 minutes** (coordinated directly between HR and On-Call Security).
- **Hardware Return:** Departing employees must return company laptops, hardware security keys, and office access badges within **3 business days** of their final working day.
