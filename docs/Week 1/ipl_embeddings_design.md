# IPL Match Predictor — Embeddings Design Summary

## Objective
Design a robust embedding system for batters, bowlers, and venues that captures **player identity, current form, matchup effects, and venue characteristics** while integrating smoothly with the IPL match simulator (wicket model → runs model → LSTM → innings simulator → bowler selection).

---

# Key Design Principle

Embeddings must represent:

**identity + current form + role + matchup + venue context + time awareness**

Not just player ID or career average.

Static career embeddings are avoided because player performance changes over time.

---

# Final Embedding Types

## 1. Player Base Embedding

### Purpose
Represents overall player identity and long-term skill.

### Input

- player_id
- historical performance patterns

### Output

```
player_base_embedding (16-dim recommended)
```

### Usage

Used in all models:

- wicket model
- runs model
- LSTM
- bowler selection

---

# 2. Batter Embedding

### Purpose
Represents batting behavior and style.

### Features

- strike rate
- batting average
- boundary %
- spin vs pace performance
- powerplay performance
- middle overs performance
- death overs performance
- dot ball %
- dismissal types

### Output

```
batter_embedding (16-dim)
```

### Used in

- runs prediction
- wicket prediction
- LSTM sequence

---

# 3. Bowler Embedding

### Purpose
Represents bowling style and effectiveness.

### Features

- economy
- wicket rate
- dot ball %
- death overs economy
- powerplay performance
- yorker %
- spin/pacer effectiveness
- boundary conceded %

### Output

```
bowler_embedding (16-dim)
```

### Used in

- wicket model
- runs model
- bowler selection
- LSTM

---

# 4. Form-Based Dynamic Embedding

### Purpose
Capture **current player form**.

### Window

Recommended:

```
last 10 matches
+ current season stats
```

### Example Features

- last 10 match economy
- last 10 match wickets
- last 10 match strike rate
- current season performance
- recent boundary rate

### Output

```
recent_form_vector (6–8 features)
```

### Why Needed

Solves career-average problem.

Captures:

- peak vs decline
- current performance
- momentum
- recent improvements

---

# 5. Venue Embedding (Static)

### Purpose
Capture pitch and ground behavior.

Venue is relatively stable, so static embedding is acceptable.

### Features

- average score
- spin wickets %
- pace wickets %
- boundary %
- average runs per over
- pitch behavior

### Output

```
venue_embedding (6–8 dim)
```

### Example

```
Chinnaswamy → high scoring
Chepauk → spin friendly
Wankhede → batting friendly
```

---

# 6. Batter–Bowler Interaction Embedding

### Purpose
Capture matchup effects.

Cricket is matchup driven.

### Construction

```
interaction_embedding = batter_embedding ⊙ bowler_embedding
```

(element-wise multiplication)

### Captures

- batter vs spin
- batter vs swing
- bowler vs left/right hand
- matchup dominance

### Output

```
interaction_embedding (16-dim)
```

---

# Final Input Structure

## Wicket Model

```
batter_embedding
bowler_embedding
interaction_embedding
venue_embedding
recent_form
match_state
```

---

## Runs Model (First 6 Balls)

```
batter_embedding
bowler_embedding
interaction_embedding
venue_embedding
recent_form
match_state
```

---

## LSTM Model

```
batter_embedding
bowler_embedding
interaction_embedding
venue_embedding
recent_form
season_context
ball_sequence
match_state
```

---

## Bowler Selection Model

```
bowler_embedding
recent_form
venue_embedding
match_state
phase
```

---

# Form Window Decision

| Window | Verdict |
|------|--------|
| last 5 matches | too noisy |
| last 10 matches | best balance |
| last 15 matches | stable but slow |
| whole season | outdated |

Final choice:

```
last 10 matches + season stats
```

---

# Embedding Dimensions

| Embedding | Dimension |
|-----------|-----------|
| player base | 16 |
| batter | 16 |
| bowler | 16 |
| venue | 6–8 |
| recent form | 6–8 |
| interaction | 16 |

Keep embeddings small to prevent overfitting.

---

# Data Flow

```
ball-by-ball data
      ↓
player stats extraction
      ↓
recent form calculation
      ↓
embedding generation
      ↓
models (wicket + runs + LSTM)
      ↓
innings simulation
```

---

# Design Decisions Summary

| Component | Decision |
|-----------|---------|
| player embedding | form based |
| batter embedding | separate |
| bowler embedding | separate |
| venue embedding | static |
| form window | last 10 matches + season |
| interaction embedding | included |
| static career embedding | avoided |

---

# Final Architecture Principle

**Player identity remains stable, form changes dynamically, venue remains static, and matchups drive outcomes.**

This ensures the simulator remains realistic, scalable, and industry-grade.

