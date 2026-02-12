# Model Autopsy

## Problem Definition

Goal: Predict cricket match outcomes using structured ball-by-ball data.

The current formulation treats prediction as a supervised regression task, leveraging match state features such as teams, venue, overs, and wickets to estimate final outcomes.


---

## Dataset Semantics

Each row represents the match state at a specific ball.

Example state:
Team A vs Team B  
Over 2, Ball 2  
Score: 15/1  

The dataset captures progressive match states rather than isolated match summaries.

However, the current modeling approach does not leverage temporal ordering between rows. Each sample is treated independently.


---

## Sequence Modeling Audit

The original architecture used an LSTM under the assumption that cricket matches are inherently sequential.

Inspection revealed that the input tensor was reshaped as:

(samples, 1, features)

This results in a sequence length of one timestep.

### Implication
The model cannot learn temporal dependencies between consecutive balls and therefore cannot benefit from recurrent memory.

### Conclusion
The LSTM effectively behaves like a feedforward network while introducing unnecessary computational complexity.


---

## Baseline Model Comparison

To validate the modeling strategy, several tree-based models were trained.

### Results

| Model | R² |
|--------|------|
| XGBoost | **0.786** |
| Random Forest | 0.746 |
| Linear Regression | 0.735 |
| Decision Tree | 0.738 |
| LSTM | ~0.745 |

### Interpretation

XGBoost significantly outperformed the LSTM baseline, indicating that the task behaves primarily as a structured tabular regression problem rather than a temporal sequence modeling problem.

### Decision

Gradient boosting models will be prioritized moving forward unless a true sequential formulation is introduced.


---

## Leakage Audit

- The `winner` column exists in the dataset but is not used during training.
- Final match outcomes are not directly fed as features.

### Risk Areas Identified

- Inclusion of late-over match states may reduce prediction difficulty.
- Temporal validation has not yet been implemented.
- Random train-test splitting may introduce subtle chronological leakage.

### Next Step
Adopt time-aware validation strategies to ensure realistic evaluation.


---

## Feature Strength Analysis

### Current Features

- batting_team  
- bowling_team  
- venue  
- over  
- ball  

### Limitations

These features do not capture team strength, player quality, or strategic context.

As a result, the model assumes identical scoring potential across teams with vastly different batting lineups.

### Planned Feature Improvements

Future iterations will prioritize engineered aggregate features such as:

- Team batting ratings  
- Bowling economy metrics  
- Historical run rates  
- Venue scoring patterns  
- Phase-of-play indicators (powerplay, middle overs, death overs)

This approach preserves signal strength without introducing excessive dimensionality from player-level encoding.


---

## Modeling Risk Assessment

### Primary Risk
The model may rely heavily on match progress indicators such as overs and wickets, making predictions easier during later stages of a match.

### Implication
Performance may degrade when predicting outcomes early in the innings.

### Mitigation Plan

- Implement phase-wise evaluation.
- Measure performance separately across powerplay, middle overs, and death overs.
- Introduce time-aware validation.


---

## Architecture Finding

The original neural approach added complexity without measurable performance gains.

Tree-based methods demonstrated superior accuracy while requiring significantly less operational overhead.

### Engineering Principle

Model complexity must be justified by measurable performance gains.  
When simpler models outperform deep architectures, they should be preferred for both reliability and operational efficiency.


---

## Strategic Pivot

The system will transition toward a gradient-boosted architecture as the primary prediction engine.

Neural approaches may be revisited if a true sequence-based simulator is developed in future iterations.


---

## Future Modeling Tracks

Two modeling paths have been identified:

### 1. Tabular Outcome Engine (Primary Production Path)

Focused on reliability, interpretability, and deployment readiness.

Capabilities will include:

- Final score prediction  
- Win probability estimation  
- Expected runs remaining  
- Match state evaluation  

This engine will form the production backbone of the platform.


### 2. Sequential Match Simulator (Research Track)

An advanced modeling initiative aimed at capturing ball-by-ball transitions.

Potential capabilities:

- Simulating innings trajectories  
- Scenario analysis  
- Strategic forecasting  
- Dynamic win probability updates  

The simulator will be developed as an experimental extension after the production system is stabilized.


---

## Immediate Action Items

- Promote XGBoost as the baseline production model.
- Separate training from inference.
- Implement dataset versioning.
- Introduce temporal validation.
- Expand feature set with strength-based metrics.
- Establish a reproducible training pipeline.
