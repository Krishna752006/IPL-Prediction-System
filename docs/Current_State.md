# Current System State

> **Last updated:** Week 1 complete (March 2026)
> **Status:** Microservice scaffold in place. ML baseline benchmarked. Training pipeline in progress.

---

## Repository Structure

```
IPL-Prediction-System/
├── ml-service/                   ✅ Active development
│   ├── src/
│   │   ├── baselines.py          ✅ Baseline comparison (XGBoost, RF, DT, LR, LSTM)
│   │   ├── simple_train_test.py  ✅ LSTM train/test runner
│   │   ├── latency_test.py       ✅ Latency benchmarking across models
│   │   └── model.py              ⚠️  Old LSTM code (legacy, kept for reference)
│   ├── models/                   ✅ Directory exists, populated during training
│   ├── data/                     ✅ Dataset v1
│   ├── main.py                   ⚠️  FastAPI skeleton (/health only)
│   └── requirements.txt
│
├── data-service/                 ✅ Pipeline scaffolded
│   ├── pipeline.py               ✅ End-to-end data pipeline
│   ├── features.py               ✅ Feature engineering (v1)
│   ├── split.py                  ✅ Train/test split logic
│   ├── ingest.py                 ✅ Data ingestion
│   └── config.py
│
├── frontend/                     ✅ Active (React + TypeScript + Vite + Tailwind)
│   └── src/                      ✅ Routes and UI components exist
│
├── api-gateway/                  🔲 Empty scaffold
├── analytics-service/            🔲 Empty scaffold
├── docs/                         ✅ Project documentation
├── Main Project/                 ⚠️  Original LSTM project (legacy — do not modify)
└── Original Data/                ✅ Raw source data
```

**Legend:** ✅ Exists and functional · ⚠️ Exists but needs attention · 🔲 Planned, not started

---

## Dataset

### Overview
- **Source:** Ball-by-ball + match-level IPL data
- **Size:** ~260,000 rows
- **Structure:** Each row = match state at a specific ball
- **Current version:** Dataset v1 (basic features only)
- **Location:** `ml-service/data/`

### Features (v1)
|   Type    |                       Features                            |
|-----------|-----------------------------------------------------------|
| Inputs    | `batting_team`, `bowling_team`, `venue`, `over`, `ball`   |
| Targets   | `runs`, `wickets`, `winner` (derived)                     |

### Known Issues
- Possible data leakage from match joins
- No time-aware train/test split (random split currently used)
- No team strength or player features
- Late-innings predictions are easier — creates evaluation bias
- Dataset v2 (engineered features) planned for Week 2–3

---

## ML Models

### Baseline Comparison (Week 1 Result)

| Model             | MSE    | MAE    | RMSE   | R²        | Latency / sample (ms) |
|-------------------|--------|--------|--------|-----------|------------------------|
| XGBoost           | 119.35 | 5.97   | 10.92  | **0.786** | 0.00238                |
| Random Forest     | 149.05 | 6.67   | 12.21  | 0.746     | 0.00806                |
| Decision Tree     | 151.64 | 6.59   | 12.31  | 0.738     | 0.00035                |
| Linear Regression | 147.40 | 6.72   | 12.14  | 0.735     | 0.00015                |
| LSTM (original)   | ~0.25* | ~0.36* | ~0.50* | ~0.745    | 290.00                 |

> *LSTM metrics used a different scale — not directly comparable. The original LSTM used sequence_length=1, making it functionally a feedforward network with 290ms latency overhead.

### Decision
**XGBoost is the chosen model going forward.** Best R² (0.786), sub-millisecond latency, and interpretable feature importance. LSTM has been deprecated.

---

## Backend

### Current State
- **Framework:** FastAPI (migrated from Flask)
- **Location:** `ml-service/main.py`
- **Working endpoints:** `/health` only
- **Pending:** `/predict` and `/win-probability`

### Issues
- No model loading or inference logic yet
- No request validation
- No error handling
- No authentication
- No logging

---

## Frontend

### Current State
- **Stack:** React + TypeScript + Vite + Tailwind CSS
- **Location:** `frontend/`
- **Routes:** Home, Login/Register (UI only), Predictions, Statistics, Team Analysis
- **API integration:** Partial — some real data connected, ML service not yet wired

### Issues
- Auth is UI-only (no backend integration)
- Prediction page not yet connected to `/predict`

---

## Infrastructure

| Component        | Status         |
|------------------|----------------|
| Docker           | 🔲 Not started |
| CI/CD            | 🔲 Not started |
| MLflow tracking  | 🔲 Not started |
| Cloud deployment | 🔲 Not started |
| Database         | 🔲 Not started |
| Monitoring       | 🔲 Not started |

---

## Key Technical Debt

### Critical
- Training and inference are not yet separated
- No model persistence or versioning
- Encoders refitted on every run (no artifact saving)
- No dataset versioning

### Medium
- No database
- No auth implementation
- No logging infrastructure

### Low
- No tests
- No rate limiting

---

## Core Insight

> The biggest limitation is **not the model** — it is data quality, architecture design, and engineering maturity.
> XGBoost on Dataset v1 already beats the original LSTM. The real performance gains will come from Dataset v2 (feature engineering) and a clean, reproducible training pipeline.
