# IPL Cricket Intelligence Platform — To Do List

> **Legend:** ✅ Done | 🔜 Planned | ⚠️ Open (not yet decided)

## 1. Architecture & System Design

- ✅ Microservices architecture
- ✅ API Gateway service
- ✅ ML Inference service (skeleton)
- ✅ Frontend service (skeleton)
- ✅ System architecture documentation
- ✅ Monorepo structure
- ✅ Service ownership separation (ML Engineer / Web Developer)
- 🔜 Data ingestion service
- 🔜 Analytics service
- 🔜 Database service
- 🔜 System architecture diagram
- 🔜 Data flow diagram
- 🔜 Deployment architecture diagram
- ⚠️ Cache service — Redis decision pending

## 2. Backend Engineering

- ✅ FastAPI for ML service
- ✅ Swagger / OpenAPI docs (auto-generated)
- ✅ Request validation (Pydantic)
- ✅ Response schemas (Pydantic)
- ✅ Structured logging
- ✅ Health check endpoints
- ✅ Environment configuration
- 🔜 Node.js API Gateway (full implementation)
- 🔜 REST API endpoints (real predictions)
- 🔜 Error handling middleware
- 🔜 Rate limiting
- 🔜 JWT authentication
- 🔜 Role-based access control (Free vs $10 tier gating)
- 🔜 Secrets management
- 🔜 Centralized config system

## 3. Frontend Engineering

- ✅ React (skeleton)
- 🔜 Component-based architecture
- 🔜 Dashboard layout
- 🔜 Match prediction UI
- 🔜 Win probability visualization
- 🔜 Analytics charts
- 🔜 Authentication pages
- 🔜 API integration layer
- 🔜 State management
- 🔜 Responsive design
- 🔜 Error boundaries
- 🔜 Form validation
- 🔜 UI component library
- 🔜 Chart library
- 🔜 Routing system
- 🔜 Subscription UI (Free vs $10 tier)
- 🔜 Payment integration UI
- 🔜 Ball-by-ball simulation output display
- 🔜 Production build optimization

## 4. Data Engineering

- ✅ Dataset v1 pipeline (CSV → Parquet)
- ✅ Feature engineering pipeline (v1)
- ✅ Time-based train/test split
- ✅ Data validation (basic)
- ✅ Missing value handling
- ✅ Feature scaling
- ✅ One-hot encoding (v1)
- ✅ Data versioning (metadata JSON per version)
- ✅ Data pipeline scripts
- ✅ Dataset storage (Parquet)
- 🔜 Dataset v2 — embeddings replacing one-hot encoding
- 🔜 Player statistics dataset
- 🔜 Venue statistics dataset
- 🔜 Outlier handling
- 🔜 Feature selection
- 🔜 Data schema documentation (`docs/feature_spec.md`)
- 🔜 Data quality checks
- ⚠️ Multiple external data sources — not yet decided

## 5. ML Engineering

- ✅ Baseline model comparison (7 models — GRU, LSTM, XGBoost, RNN, LR, RF, DT)
- ✅ XGBoost model
- ✅ RandomForest model
- ✅ LSTM model
- ✅ Feature importance analysis (basic)
- ✅ Model evaluation metrics (RMSE, MAE, R², Adj. R², latency)
- ✅ Model comparison (full leaderboard)
- ✅ Model serialization (.pkl bundles)
- ✅ Model configuration (YAML-driven)
- ✅ Experiment tracking (MLflow)
- ✅ Reproducible training pipeline
- 🔜 Player embedding system (batter, bowler, base, venue)
- 🔜 Wicket prediction model
- 🔜 First 6 balls runs model
- 🔜 LSTM or GRU innings sequence model (decided by results)
- 🔜 Match simulation engine
- 🔜 RL bowler selection model
- 🔜 Win probability output (from simulation)
- 🔜 Score prediction output (from simulation)
- 🔜 Cross validation
- 🔜 Hyperparameter tuning
- 🔜 Probability calibration
- 🔜 Inference contract (formal input/output schema for prediction API)
- 🔜 Inference logging (log predictions + latency per request)
- 🔜 Performance benchmarking
- 🔜 End-to-end simulation validation (outputs produce realistic IPL scores)
- ⚠️ LightGBM model — not yet decided if needed

## 6. MLOps

- ✅ MLflow experiment tracking
- ✅ Model registry (staging → production → history)
- ✅ Dataset versioning
- ✅ Training pipeline (reproducible, config-driven)
- ✅ Training logs
- ✅ Model lifecycle management (staging → production → history)
- ✅ Artifact storage (MLflow artifacts)
- 🔜 Inference pipeline (production-grade)
- 🔜 Retraining pipeline
- 🔜 Automated evaluation
- 🔜 Model version control (full lifecycle)
- 🔜 Pipeline documentation
- 🔜 Drift detection

## 7. Database & Storage

- ✅ Parquet storage for datasets
- ✅ MLflow artifact storage (models, configs, metadata)
- 🔜 Embeddings storage & retrieval
- 🔜 User data storage
- 🔜 Player stats storage
- 🔜 Match history storage
- 🔜 Team squads & batting/bowling order storage
- 🔜 Schema design
- 🔜 Data access layer
- 🔜 Indexing
- 🔜 Backup strategy
- 🔜 Cloud storage for models
- 🔜 Cloud storage for datasets
- ⚠️ MongoDB vs PostgreSQL — not yet decided

## 8. Caching

- 🔜 Redis setup
- 🔜 Prediction caching (cache simulation results)
- 🔜 Analytics caching

## 9. Subscription & Payments

- 🔜 Free tier feature gating
- 🔜 $10 lifetime tier feature gating
- 🔜 Payment integration (full)
- 🔜 User account system (linked to tier)
- 🔜 Access control per feature (simulation, custom teams, season sim)

## 10. DevOps & Infrastructure

- ✅ Docker for all services
- ✅ Docker Compose (multi-container)
- ✅ Environment variables
- ✅ CI/CD with GitHub Actions
- ✅ Build verification (lint + format checks)
- 🔜 Automated testing in CI
- 🔜 Deployment pipeline
- 🔜 Cloud hosting (AWS / Render)
- 🔜 Domain setup
- 🔜 HTTPS
- 🔜 Static frontend hosting
- 🔜 Kubernetes deployment

## 11. Monitoring & Reliability

- ✅ Structured logging (file + console, environment-aware)
- ✅ Log rotation (7-day, TimedRotatingFileHandler)
- 🔜 Centralized observability (standardized logs across services)
- 🔜 Distributed tracing (end-to-end request tracking)
- 🔜 Application monitoring (latency, errors, uptime)
- 🔜 System health checks (per service status + readiness)
- ⚠️ Alerts — not yet decided
- ⚠️ Monitoring dashboard — not yet decided

## 12. Security

- ✅ Pre-commit hooks (code quality enforcement)
- 🔜 JWT authentication
- 🔜 Password hashing
- 🔜 Role-based access (Free vs $10 tier)
- 🔜 Rate limiting
- 🔜 CORS policy
- 🔜 Secrets management (no hardcoded keys, env-based)
- 🔜 HTTPS
- 🔜 Secure headers
- 🔜 Input validation (all endpoints)

## 13. Testing

- ✅ ML unit tests (pipeline, bundle, inference, registry)
- ✅ Data pipeline tests
- ✅ Inference tests
- ✅ CI test automation (GitHub Actions)
- 🔜 Backend unit tests (service logic, business rules, auth flows)
- 🔜 API integration tests (end-to-end request/response validation)
- 🔜 End-to-end system tests (full user journey across components)
- 🔜 Contract tests (service-to-service / frontend-backend compatibility)
- 🔜 Load / performance tests (latency, throughput, scaling limits)
- 🔜 Failure-mode / resilience tests (timeouts, partial outages, retries)
- 🔜 Security tests (auth, injection, data exposure checks)
- 🔜 Test suite runner (executes multiple unit tests together)
- 🔜 Coverage reports (threshold-based visibility and enforcement)

## 14. Documentation

- ✅ README
- ✅ Architecture documentation (written)
- ✅ ML pipeline docs (Week 1 report)
- 🔜 `docs/feature_spec.md` — formal feature schema (**gap from Week 1, due before Week 2/3**)
- 🔜 Architecture diagram — visual flow (**gap from Week 1, due before Week 2/3**)
- 🔜 API documentation
- 🔜 Data pipeline docs
- 🔜 Deployment guide
- 🔜 Developer setup guide
- 🔜 Model explanation docs
- 🔜 Project vision document

## 15. Product & Recruiter Signal

- ✅ Clean GitHub repo
- ✅ Proper commits
- 🔜 Live hosted URL
- 🔜 Demo video
- 🔜 Architecture diagrams
- 🔜 Issue tracking
- 🔜 Feature roadmap
- 🔜 Blog article
- 🔜 Resume bullets
- 🔜 Screenshots
- 🔜 Performance metrics writeup
- 🔜 System design explanation

## 16. Team Workflow

- ✅ Git branching strategy
- ✅ Feature branches
- ✅ Pull requests
- ✅ Code reviews
- ✅ Service ownership (ML Engineer / Web Developer)
- 🔜 Task tracking
- 🔜 Weekly milestones
- 🔜 Issue management
