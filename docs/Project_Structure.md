# Project Structure Reference

> This file exists to give an AI assistant (or new contributor) immediate orientation.
> Always consult this before creating or modifying files.

---

## Root

```
IPL-Prediction-System/
├── ml-service/           ML model, training pipeline, inference API
├── data-service/         Data ingestion, feature engineering, dataset versioning
├── frontend/             React + TypeScript user interface
├── api-gateway/          Request routing between services (not yet implemented)
├── analytics-service/    Team/venue/match analytics endpoints (not yet implemented)
├── docs/                 Project documentation (this folder)
├── Main Project/         ⚠️ LEGACY — original LSTM project. Do not modify.
├── Original Data/        Raw source IPL data files
└── .gitignore
```

---

## ml-service/

**Purpose:** Everything ML — training, model artifacts, and inference API.

```
ml-service/
├── src/
│   ├── baselines.py          Compares XGBoost, RF, DT, Linear Regression, LSTM
│   ├── simple_train_test.py  Standalone XGBoost train + evaluate script
│   ├── latency_test.py       Benchmarks prediction latency per model
│   └── model.py              ⚠️ Legacy LSTM model code — do not use for new work
├── models/
│   ├── (xgboost_v1.pkl)      Trained model artifact — saved here after training
│   └── encoders/             Label encoders and scalers — saved here after training
├── data/
│   └── (dataset files)       Dataset v1 CSVs live here
├── main.py                   FastAPI app — currently /health only
├── models.py                 Pydantic request/response schemas
└── requirements.txt
```

**What exists:** Training scripts, baseline benchmarks, latency tests, FastAPI skeleton.
**What's missing:** `/predict` endpoint, model loading logic, saved artifacts, MLflow integration.

**When adding new files here:**
- Training code → `src/train.py`
- Inference logic → `src/predict.py`
- New endpoints → `main.py`
- Saved models → `models/`
- New datasets → `data/`

---

## data-service/

**Purpose:** Data ingestion, cleaning, feature engineering, and dataset versioning.

```
data-service/
├── pipeline.py     Orchestrates the full data pipeline (ingest → features → split)
├── features.py     Feature engineering functions (v1: basic team/venue/over features)
├── split.py        Train/test split — needs time-aware split implementation
├── ingest.py       Loads raw data from Original Data/
└── config.py       File paths and configuration constants
```

**What exists:** Full pipeline scaffold with working v1 feature engineering.
**What's missing:** Time-aware split, Dataset v2 features (team strength, rolling averages, venue patterns).

**When adding Dataset v2 features:** Add to `features.py` under a clearly named function. Do not break v1 functions — keep them for comparison.

---

## frontend/

**Purpose:** React user interface.

```
frontend/
├── src/            Components, pages, and routing
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

**Stack:** React + TypeScript + Vite + Tailwind CSS.
**What exists:** Routes for Home, Login/Register (UI only), Predictions, Statistics, Team Analysis.
**What's missing:** Real auth, wired `/predict` call on Predictions page, analytics endpoint integration.

**Note:** Do not add new pages until the corresponding backend endpoint exists.

---

## api-gateway/

**Purpose:** Single entry point that routes requests to ml-service, data-service, analytics-service.
**Current state:** Empty scaffold. Do not add logic here until Phase 3.

---

## analytics-service/

**Purpose:** Serve team stats, venue stats, historical match data.
**Current state:** Empty scaffold. Do not add logic here until Phase 3.

---

## docs/

**Purpose:** Project documentation for AI assistants, contributors, and recruiters.

```
docs/
├── Current_State.md      What exists right now, with repo structure and known issues
├── Goal.md               Target architecture, success criteria, tech stack
├── Roadmap.md            Phase-by-phase plan with checkboxes
└── Project_Structure.md  This file — repo map and file placement rules
```

---

## Legacy Warning

`Main Project/` contains the original LSTM-based project built before the architectural redesign. It is kept for reference only. Do not copy patterns from it — the training/inference coupling, Flask setup, and LSTM model are all being replaced.

---

## Key Conventions

- All new Python services use **FastAPI**, not Flask
- Model artifacts are always saved to `ml-service/models/` — never committed raw to git
- Dataset files live in `ml-service/data/` — raw source files stay in `Original Data/`
- Frontend never calls ml-service directly — all calls go through api-gateway (once built)
- Feature engineering changes go in `data-service/features.py`
- MLflow runs are stored locally during development
