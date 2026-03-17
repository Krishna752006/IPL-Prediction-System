# Model Design — IPL Intelligence Platform

## 1. Problem Definition

### Primary Task
Live match **win probability prediction** during an IPL match.

### Secondary Task
Final **score prediction** for both teams.

### Prediction Modes
- **Pre-match (Basic Tier)**  
  Input: teams, venue  
  Output: initial win probability + predicted scores

- **Live Match (Core Feature)**  
  Input: current match state (score, overs, wickets, etc.)  
  Output: dynamic win probability updated over time

---

## 2. Modeling Approach

### Model Selection

| Task | Model |
|------|------|
| Win Probability | LightGBM Classifier |
| Score Prediction | LightGBM Regressor |
| Baseline | Logistic Regression |

### Rationale

- Data is **tabular with engineered features** → tree-based models 'can' outperform deep learning
- Requires **low-latency inference** for API usage
- LightGBM provides:
  - High accuracy on structured data
  - Fast inference
  - Feature importance for explainability

## 3. Input Features

### A. Static Features
- Team 1
- Team 2
- Venue

### B. Pre-Match Features
- Team strength
- Player-level aggregates
- Venue run bias

### C. Live Match Features (Core)
- Current score
- Overs completed
- Wickets lost
- Current run rate
- Target score (if chasing)
- Required run rate

### D. Derived Features
- Momentum (last N overs)
- Match phase (powerplay / middle / death)
- Pressure index (RR vs required RR gap)

---

## 4. Model Outputs

### Win Probability API Output

```json
{
  "team1_win_prob": 0.64,
  "team2_win_prob": 0.36
}
````

### Score Prediction API Output

```json
{
  "predicted_score_team1": 178,
  "predicted_score_team2": 165
}
```

---

## 5. Evaluation Strategy

### Win Probability

* Log Loss (primary)
* ROC-AUC
* MCC

### Score Prediction

* RMSE (primary)
* R_squared

### Validation Method

* **Time-based split (season-wise)**
* Performance evaluated per season

---

## 6. Inference Design

### Pre-Match Flow

User Input → (teams, venue)
→ Feature Enrichment (team stats, venue data)
→ Model Prediction

### Live Match Flow

Live Data → Match State Reconstruction
→ Feature Engineering
→ Model Prediction (real-time)

---

## 7. Key Design Decisions

* Prioritized **live win probability** as core differentiator
* Separated **classification and regression tasks**
* Designed for **API-first inference**

---

## 8. Future Improvements

* Model calibration (Platt scaling / isotonic)
* Real-time streaming pipeline
* Drift detection and retraining
* Player-level contextual embeddings