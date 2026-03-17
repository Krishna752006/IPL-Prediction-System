# IPL Win Probability & Score Prediction — Data & Feature Strategy (Week 1)

## 🎯 Objective

Build a **ball-by-ball state-based dataset** to support:

- **Primary Task:** Win Probability (classification)
- **Secondary Task:** Score Prediction (regression)

---

## 🧠 Core Approach

- Convert ball-by-ball data into **independent state rows**
- Each ball = one training sample

---

## 📌 Training Sample Definition

At each ball:

X = {
  batting_team,
  bowling_team,
  venue,
  toss_winner,
  toss_decision,

  runs,
  wickets,
  overs,
  balls_remaining,
  current_run_rate,

  required_runs (if chasing),
  required_run_rate (if chasing),

  last_5_avg_score,
  last_5_avg_wickets,
  last_5_economy,

  avg_first_innings_score,
  avg_second_innings_score
}

Targets:

y_win = 1 if batting team wins else 0  
y_score = final innings score  

---

## ⚙️ Dataset Granularity

- **Ball-by-ball state-based dataset**
- Each match generates ~200+ rows
- Enables live win probability prediction

---

## 🧩 Feature Set

### Tier 1 (Must Implement)

**Match Context**
- batting_team
- bowling_team
- venue
- toss_winner
- toss_decision

**Match State**
- runs
- wickets
- overs
- balls_remaining
- current_run_rate

---

### Tier 2 (Must Implement)

**Chasing Features**
- required_runs
- required_run_rate

**Team Strength (rolling, last 5 matches)**
- avg_score
- avg_wickets
- economy

**Venue Stats**
- avg_first_innings_score
- avg_second_innings_score

---

## 🏗️ Dataset Creation Pipeline

### Step 1 — Iterate Matches

- Load match data
- Determine winner
- Split into innings

---

### Step 2 — Iterate Ball-by-Ball

For each ball:

- Update:
  - runs
  - wickets
  - overs
  - balls_remaining

- Compute:
  - current_run_rate
  - required_runs (if chasing)
  - required_run_rate (if chasing)

---

### Step 3 — Attach Static Features

- Team rolling stats (precomputed)
- Venue averages

---

### Step 4 — Assign Targets

- y_win = 1 if batting team wins else 0
- y_score = final innings score

---

### Step 5 — Store Row

- Append to dataset
- Convert to DataFrame
- Save as:

data/processed/v2_alpha.parquet

---

## 🔀 Train / Validation / Test Split

- Train: 2016–2023
- Validation: 2024
- Test: 2025

---

## 🚀 Deliverables

- Script:
  data-service/build_dataset.py

- Output dataset:
  data/processed/v2_alpha.parquet

- Basic validation:
  - shape
  - columns
  - sample rows