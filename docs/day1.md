# Day 1 — System Autopsy

## Objective
Understand the legacy IPL prediction system before making any architectural or modeling changes.

No refactoring, tuning, infra work, or feature building was performed intentionally.

---

## Timeline
Project deadline extended to **mid-April** to allow production-grade engineering and modeling decisions.

---

## System Snapshot
- Backend: Flask  
- ML: TensorFlow LSTM (legacy) + tree baselines  
- Frontend: React  
- Dataset: Ball-by-ball cricket data  

System runs locally end-to-end.

---

## Environment
- Python: 3.12.10  
- OS: Windows 11  
- CPU: Intel64  
- GPU: Not detected (CPU only)

Environment recreated successfully. Training and inference verified.

---

## Model Findings

### Legacy Assumption
LSTM chosen because cricket is sequential.

### Critical Discovery
Sequence length = **1**

Result:  
LSTM behaves like a feedforward model with unnecessary complexity.

---

## Features
- batting_team (one-hot)  
- bowling_team (one-hot)  
- venue (one-hot)  
- over  
- ball  

Targets:
- runs  
- wickets  

`winner` exists but is not modeled directly.

---

## Baseline Result

| Model | R² |
|------|------|
| XGBoost | **0.786** |
| Random Forest | 0.746 |
| Linear Regression | 0.735 |
| Decision Tree | 0.738 |
| LSTM | ~0.745 |

### Key Insight
The problem currently behaves as a **tabular regression task**, not a sequential one.

Production direction will favor gradient boosting unless a true sequential simulator is built.

---

## Dataset Notes
~260k rows, ball-level match state.

### Strengths
- High resolution  
- Progressive scoreboard  
- Simulation potential  

### Weaknesses
- No player skill features  
- No team strength metrics  
- Simplistic encoding  

### Modeling Risk
Current features assume uniform batting capability across teams — unrealistic.

### Prediction Horizon Risk
Late-innings states are easier to predict than early-game scenarios.  
Future validation must segment by match phase.

---

## Major Insight
Predicting runs ≠ predicting match outcomes.

**Win probability** will be the primary production objective.

---

## Future Dataset Direction
A new dataset will be engineered using accumulated domain and ML experience.

Focus areas:

- Team strength metrics  
- Player aggregates (without exploding dimensionality)  
- Venue behavior  
- Historical run rates  
- Phase-of-play features  

Dataset quality is expected to drive larger gains than model complexity.

---

## Technical Debt Identified
- Training coupled with API  
- No dataset versioning  
- Model overwritten without evaluation  
- Preprocessing artifacts not saved  
- No tests / logging / CI  
- Neural model selected before baseline validation (now corrected)

Debt documented — not resolved today.

---

## Decisions
- Do NOT refactor yet  
- Do NOT migrate frameworks  
- Do NOT tune models  

Clarity before rebuilding.

---

## Emerging Architecture Direction

### Production Path
**Tabular Outcome Engine**
- reliable  
- interpretable  
- deployable  

### Research Path
**Sequential Match Simulator**
- scenario forecasting  
- ball-by-ball modeling  

---

## True Sequential Modeling Plan

The original LSTM failed because it was not given real sequences.

A future research track will introduce **true ball-to-ball sequence modeling**, where each prediction is conditioned on prior deliveries within the innings.

Potential capabilities:

- Dynamic match state transitions  
- Scenario simulation  
- Real-time win probability updates  
- Strategy forecasting  

This model will only be pursued after the production tabular engine is stabilized.

Deep learning will be applied where it provides structural advantage — not by default.

The goal is not to use neural networks unnecessarily, but to deploy them where temporal dependency creates modeling leverage.

---

## Day 1 Result
Ambiguity removed.  
System understood.  
Risks identified.  
Direction clarified.

Project ready for structured engineering work.

---

# Context for Next ChatGPT

You are advising on the rebuild of an IPL prediction system after a completed technical autopsy.

Key facts:

- Legacy LSTM ineffective (sequence length = 1)
- XGBoost is current best baseline
- Dataset is ball-by-ball (~260k rows)
- Feature space is weak
- Training is tightly coupled to inference
- Technical debt documented but unresolved
- No refactoring has started
- Deadline: mid-April
- Goal: production-grade ML system

Prioritize:

- validation strategy  
- feature engineering  
- production ML practices  
- system design  

Avoid premature infra or unnecessary complexity.

Act as a senior ML engineer guiding a serious production project.

---

# Web / System Lead — Day 1
*(To be completed by teammate)*

[Teammate will paste their Day 1 findings here]
