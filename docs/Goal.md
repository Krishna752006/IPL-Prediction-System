# Project Goal

## What This Project Is

A production-grade cricket intelligence platform for IPL match prediction and analytics — built to demonstrate end-to-end ML engineering: from raw data pipelines to a hosted, API-driven product with a real frontend.

This is not a research notebook. The goal is a deployable system with clean separation between training, inference, and serving layers.

---

## Primary Objective

Build a subscription-tier IPL analytics platform that:
- Predicts match outcomes (win probability + score) in real time
- Serves predictions via a low-latency REST API
- Presents insights through an interactive analytics dashboard
- Is reproducible, versioned, and deployable

---

## Target Tech Stack

|     Layer     |               Technology                      |
|---------------|-----------------------------------------------|
| ML            | XGBoost, scikit-learn, MLflow                 |
| Data pipeline | Python, pandas, custom feature engineering    |
| Backend       | FastAPI, Python                               |
| Frontend      | React, TypeScript, Vite, Tailwind CSS         |
| Infra         | Docker, GitHub Actions CI                     |
| Deployment    | Cloud-hosted (TBD: Render / Railway / GCP)    |

---

## Core Capabilities (Must-Have)

- **Win probability estimation** — ball-by-ball, calibrated output
- **Score prediction** — runs and wickets at any match state
- **Analytics dashboard** — team performance, venue patterns, match state breakdown
- **API-based inference** — clean REST endpoints, <1s latency

---

## Secondary Capabilities

- LLM-based chatbot for cricket Q&A (RAG over dataset)
- Scenario simulation engine (what-if analysis)
- Subscription-based feature tiers (free vs. premium data access)

---

## Why This Is Non-Trivial

- **Data leakage is a real risk** — match joins must be handled carefully to avoid future information bleeding into training
- **Time-aware validation is required** — random splits overestimate performance; models must be validated on future seasons
- **Feature engineering dominates model performance** — team strength, rolling form, and venue bias require careful construction from raw ball-by-ball data
- **Inference and training must be fully decoupled** — encoders, scalers, and model artifacts must be saved and versioned separately from training code
- **Latency constraints are real** — LSTM at 290ms is unusable; XGBoost at 0.002ms is the right tradeoff

---

## Success Criteria

### ML
- Model outperforms LSTM baseline on time-aware validation
- R² > 0.8 on Dataset v2 (engineered features)
- Calibrated probability outputs (not raw scores)

### Engineering
- Clean training / inference separation
- MLflow experiment tracking and model versioning
- API latency < 1 second end-to-end
- Reproducible pipeline (data → features → model → artifact)

### Product
- Fully integrated frontend + backend
- Live hosted demo
- No dummy data — all UI elements backed by real model outputs

---

## Non-Goals

- No deep learning unless benchmarks justify it
- No Kubernetes or overengineered infra
- No betting-style or gambling-adjacent predictions
- No focus on real-time data ingestion (static dataset is sufficient for v1)

---

## Strategic Principle

> **Feature quality > Model complexity > UI aesthetics**

A well-engineered XGBoost on rich features will outperform a poorly-featured neural network — and be faster, more interpretable, and easier to deploy.
