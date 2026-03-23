# IPL Match Predictor & Simulator Architecture

## Overview

This project builds a full **IPL match simulation and prediction system** using multiple machine learning models and a reinforcement learning-based bowler selection strategy. The system uses ball-by-ball data (2008–2024) and simulates an entire innings based on player embeddings, match state, and probabilistic modeling.

The architecture is modular, allowing each component to be trained separately and integrated into a final match simulator.

---

# High-Level Architecture

```
Dataset (Ball-by-ball IPL data 2008–2024)
        |
        v
Feature Engineering + Player Embeddings
        |
        v
-----------------------------------------
|               ML MODELS               |
-----------------------------------------
| 1. Wicket Prediction Model            |
| 2. First 6 Balls Runs Model           |
| 3. LSTM Innings Model                 |
| 4. RL Bowler Selection Model          |
-----------------------------------------
        |
        v
Match Simulator Engine
        |
        v
Full IPL Match Simulation & Winner Prediction
```

---

# Core Components

## 1. Player Embedding System

### Purpose
Represent players numerically since cricket is heavily player-driven.

### Inputs
- Batter ID
- Bowler ID
- Player stats
- Historical performance
- Matchups

### Output

```
player_id → embedding vector (e.g., 32/64 dimensional)
```

### Storage
Database stores:

- batter embeddings
- bowler embeddings
- team squads
- batting order
- bowling pool

### Usage
Used in:
- wicket model
- runs model
- LSTM
- RL

---

# 2. Wicket Prediction Model

## Purpose
Predict whether a ball results in a wicket or not.

## Input Features

- bowler embedding
- batter embedding
- over number
- ball number
- runs scored
- wickets fallen
- match phase
- venue
- innings

## Output

```
0 → not out
1 → wicket
```

## Behavior

If wicket:

```
runs = 0
next batter comes from batting order
wickets += 1
```

Else:

```
move to runs prediction model
```

---

# 3. First 6 Balls Runs Model

## Purpose
Predict runs for the first over (first 6 balls) of a batter sequence.

## Why Needed

LSTM requires sequence history.

First 6 balls provide initial sequence for LSTM.

## Input

- bowler embedding
- batter embedding
- match state
- phase
- over
- ball

## Output

```
0,1,2,3,4,6
```

## Result

Creates initial sequence:

```
[1,0,4,2,1,0]
```

This goes into LSTM.

---

# 4. LSTM Innings Model

## Purpose
Model innings as a time series and predict future runs.

## Input

```
last 6 balls sequence
+ match state
+ player embeddings
+ wickets
+ overs
```

## Output

```
runs on next ball
```

## Behavior

After each ball:

- update sequence
- feed to LSTM
- predict next run


## Strike Rotation

### Odd run

```
strike rotates
```

### End of over

```
strike rotates
```

---

# 5. Match Simulator Engine

## Purpose
Control the entire innings.

## Tracks

- runs
- wickets
- overs
- balls
- strike
- batting order
- bowling order
- target

## Logic

### Ball Loop

```
choose bowler
predict wicket
if wicket:
    next batter
else:
    predict runs
update match state
update strike
update over
check innings end
```

## Innings End Conditions

### 10 wickets

```
innings over
```

### 20 overs completed

```
innings over
```

### Target chased

```
2nd team wins
```

Else

```
1st team wins
```

---

# 6. RL Bowler Selection Model

## Purpose
Choose which bowler bowls the next over.

## Agent

Captain AI

## State

```
over number
runs
wickets
batter embeddings
available bowlers
overs left
match phase
```

## Action

```
choose bowler
```

## Reward

### Option 1

```
+10 wicket
-1 per run
```

### Option 2

```
+100 win
-100 loss
```

## Training Environment

RL interacts with:

- wicket model
- runs model
- LSTM
- simulator


## Training Loop

```
state → choose bowler
simulate over
get reward
update policy
repeat
```

---

# Training Order

## Step 1

Train:

- player embeddings
- wicket model
- first 6 balls model
- LSTM model


## Step 2

Build simulator engine

Use rule-based bowler selection initially.


## Step 3

Train RL bowler model using simulator.


## Step 4

Replace rule-based with RL.


## Step 5

Full system integration.

---

# Data Flow

```
Database
   |
   v
Embeddings
   |
   v
Wicket Model
   |
   |---- wicket → next batter
   |
   v
Runs Model (first 6 balls)
   |
   v
LSTM Model
   |
   v
Match Simulator
   |
   v
RL Bowler Selection
   |
   v
Winner Prediction
```

---

# Tech Stack Suggestion

## ML

- Python
- PyTorch
- Scikit-learn
- Pandas
- NumPy

## RL

- Stable Baselines3
- PyTorch
- Gym-style environment

## Backend

- FastAPI
- PostgreSQL
- Redis

## MLOps

- Docker
- MLflow
- Airflow

## Deployment

- AWS / GCP
- Kubernetes (optional)

---

# Final System Output

## Match Simulation

```
Team A vs Team B

Team A: 178/6
Team B: 175/9

Winner: Team A
```

## Ball-by-ball commentary

```
Over 17: Bumrah to Kohli → 1 run
Over 17: Bumrah to Maxwell → 4 runs
```

## Analytics

- win probability
- expected runs
- bowler effectiveness
- batter performance

---

# Final Goal

Build a **realistic AI-driven IPL match simulation engine** with:

- player embeddings
- probabilistic ball prediction
- time series innings modeling
- RL-based bowler selection
- full match simulation

This creates an industry-grade sports analytics and simulation system.

