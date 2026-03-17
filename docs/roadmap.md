# Development Roadmap

## ✅ Phase 0 — System Audit & Baseline (Week 1) — COMPLETE

### Completed
- Audited original LSTM project (identified sequence_length=1 flaw)
- Benchmarked XGBoost, Random Forest, Decision Tree, Linear Regression, LSTM
- Established XGBoost as the primary model (R²: 0.786, latency: 0.002ms)
- Migrated project to microservice folder structure
- Scaffolded: `ml-service`, `data-service`, `frontend`, `api-gateway`, `analytics-service`
- Basic FastAPI setup in `ml-service/main.py` (`/health` endpoint)
- Data pipeline scaffolded in `data-service/` (pipeline, features, split, ingest)

---

## Phase 1 — ML Engine + Training Pipeline (Week 2)

### ML (`ml-service/`)
- [ ] Build reproducible training script (`src/train.py`)
- [ ] Implement time-aware train/test split (by season, not random)
- [ ] Save trained model artifact to `models/xgboost_v1.pkl`
- [ ] Save encoders and scalers to `models/encoders/`
- [ ] Log experiments with MLflow (local)
- [ ] Begin Dataset v2 feature engineering (see Parallel Track)

### Backend (`ml-service/main.py`)
- [ ] Implement `/predict` endpoint (load saved model + encoders)
- [ ] Implement `/win-probability` endpoint
- [ ] Add request validation (Pydantic schemas)
- [ ] Add basic error handling

### Infra
- [ ] Add `Dockerfile` for `ml-service`

---

## Phase 2 — Dataset v2 + Model Improvement (Week 3)

### Data (`data-service/`)
- [ ] Add team strength metrics (win rate, recent form)
- [ ] Add venue scoring patterns (avg runs, wicket rates by venue)
- [ ] Add rolling averages (last N matches)
- [ ] Version dataset as `data_v2.csv`

### ML (`ml-service/`)
- [ ] Retrain XGBoost on Dataset v2
- [ ] Compare v1 vs v2 metrics in MLflow
- [ ] Add phase-aware evaluation (powerplay / middle / death overs)
- [ ] Build win probability calibration

### Infra
- [ ] Set up CI pipeline (GitHub Actions: lint + test)

---

## Phase 3 — Product Layer (Week 4)

### Backend
- [ ] Build `analytics-service` endpoints (team stats, venue stats, match history)
- [ ] Wire `api-gateway` to route between services
- [ ] Add response caching (optional, for analytics endpoints)

### Frontend (`frontend/`)
- [ ] Connect Predictions page to `/predict`
- [ ] Connect Statistics page to analytics endpoints
- [ ] Build win probability visualization (ball-by-ball chart)
- [ ] Ensure no dummy data remains in any connected page

---

## Phase 4 — Advanced Features (Week 5)

### ML
- [ ] Scenario simulation (what-if: change team, venue, over)
- [ ] Pseudo-live prediction (polling simulation)

### AI
- [ ] Start LLM chatbot — RAG over match dataset
- [ ] Expose via `/chat` endpoint

---

## Phase 5 — Production Hardening (Week 6)

### Backend
- [ ] JWT authentication
- [ ] Structured logging (all services)
- [ ] Rate limiting
- [ ] Monitoring setup (basic)

### ML
- [ ] Retraining pipeline (automated on new data)
- [ ] Model versioning via MLflow model registry

---

## Phase 6 — Deployment & Polish (Week 7–8)

### Infra
- [ ] Deploy backend + frontend (Render / Railway / GCP)
- [ ] Domain + HTTPS
- [ ] End-to-end latency validation (<1s target)

### Product
- [ ] UI polish pass
- [ ] Demo mode (no auth required for showcase)

### Documentation
- [ ] Architecture diagram
- [ ] Updated README with setup instructions
- [ ] Blog post / write-up for portfolio

---

## Parallel Track — Dataset v2 (Weeks 2–4)

This is the highest-leverage work in the project. Feature quality determines model ceiling more than model choice.

Priority features:
- Team win rate (overall + last N matches)
- Head-to-head team records
- Venue-specific scoring patterns
- Toss decision impact
- Rolling run rate and wicket rate

---

## Execution Rules

- Do NOT add LLM until Phase 4 — APIs must stabilize first
- Do NOT build new UI pages until the backend endpoint exists
- Do NOT optimize models before fixing data (Dataset v2 first)
- Do NOT skip MLflow setup — reproducibility is a core deliverable

---

## Final Deliverable

- Live hosted platform with real predictions
- Microservice-style backend (FastAPI)
- Reproducible ML pipeline (data → features → model → artifact → API)
- React + TypeScript frontend fully integrated with backend
- MLflow experiment tracking
- Recruiter-grade README and architecture documentation
