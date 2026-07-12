Here are the exact formulas used in the pipeline.

---

## 1. Phase Assignment

You convert over number → phase bucket.

```text
0–5   → Powerplay
6–14  → Middle
15–19 → Death
```

Formula:

```text
phase(over)
=
{
powerplay, if over ≤ 5
middle,    if 6 ≤ over ≤ 14
death,     if over ≥ 15
}
```

---

## 2. Unique Bowler Overs

You are counting **overs**, not balls.

A bowler-over is uniquely defined as:

```text
(matchId, inning, over, bowler)
```

Formula:

```text
BowlerOvers
=
Unique(matchId, inning, over, bowler)
```

Why?

Because:

```text
1 over ≠ 1 row
```

You have ball-level data.

Example:

```text
6 deliveries by Bumrah in over 18
```

becomes:

```text
1 over
```

not:

```text
6 overs
```

---

## 3. Powerplay Overs

Formula:

```text
pp_overs(bowler)
=
count(
phase == powerplay
)
```

Mathematically:

```text
pp_overs_i
=
Σ I(phase = powerplay)
```

where:

```text
I(condition)
=
1 if true
0 otherwise
```

---

## 4. Middle Overs

Formula:

```text
middle_overs(bowler)
=
count(
phase == middle
)
```

Mathematically:

```text
middle_overs_i
=
Σ I(phase = middle)
```

---

## 5. Death Overs

Formula:

```text
death_overs(bowler)
=
count(
phase == death
)
```

Mathematically:

```text
death_overs_i
=
Σ I(phase = death)
```

---

## 6. Total Overs

Formula:

```text
total_overs
=
pp_overs
+
middle_overs
+
death_overs
```

\text{total_overs}=\text{pp_overs}+\text{middle_overs}+\text{death_overs}

Example:

```text
30 + 50 + 20 = 100
```

---

## 7. Matches Played

A match counts if player appears as:

* batsman
* non_striker
* bowler

Formula:

```text
matches
=
count unique(matchId)
```

for:

```text
player ∈ {
batsman,
non_striker,
bowler
}
```

Mathematically:

```text
matches_i
=
|Unique(matchId_i)|
```

This is:

```text
matches played
```

NOT:

```text
matches bowled
```

---

## 8. Average Overs Per Match

Formula:

```text
avg_overs_per_match
=
total_overs / matches
```

\text{avg_overs_per_match}=\frac{\text{total_overs}}{\text{matches}}

Example:

```text
total_overs = 120
matches = 40

120 / 40 = 3.0
```

Interpretation:

```text
average bowling workload
```

---

## 9. Powerplay Dominance Ratio

Formula:

```text
pp_dominance_ratio
=
pp_overs / total_overs
```

\text{pp_dominance_ratio}=\frac{\text{pp_overs}}{\text{total_overs}}

Example:

```text
pp_overs = 40
total_overs = 100

0.40
```

Interpretation:

```text
40% of bowling workload in powerplay
```

---

## 10. Middle Dominance Ratio

Formula:

```text
middle_dominance_ratio
=
middle_overs / total_overs
```

\text{middle_dominance_ratio}=\frac{\text{middle_overs}}{\text{total_overs}}

Example:

```text
middle_overs = 60
total_overs = 100

0.60
```

Interpretation:

```text
60% workload in middle overs
```

---

## 11. Death Dominance Ratio

Formula:

```text
death_dominance_ratio
=
death_overs / total_overs
```

\text{death_dominance_ratio}=\frac{\text{death_overs}}{\text{total_overs}}

Example:

```text
death_overs = 25
total_overs = 100

0.25
```

Interpretation:

```text
25% workload in death overs
```

---

## 12. Dominance Ratio Sum Property

Since:

```text
total_overs
=
pp
+
middle
+
death
```

the ratios satisfy:

```text
pp_dominance_ratio
+
middle_dominance_ratio
+
death_dominance_ratio
=
1
```

\text{pp_dominance_ratio}+\text{middle_dominance_ratio}+\text{death_dominance_ratio}=1

This is useful for sanity checking.

Example:

```text
0.20 + 0.50 + 0.30 = 1
```

If:

```text
≠ 1
```

something is wrong in aggregation.

---

## 13. Pacer Label

Formula:

```text
is_pacer
=
mode(is_pacer values)
```

Example:

```text
0
```

Interpretation:

```text
stable bowling type
```

rather than noisy per-row labeling.

---

## Final Feature Vector Per Bowler

Conceptually:

```text
[
pp_overs,
middle_overs,
death_overs,
total_overs,
matches,
avg_overs_per_match,
pp_dominance_ratio,
middle_dominance_ratio,
death_dominance_ratio,
is_pacer
]
```

These are the statistics your cold-start system will later use for:

```text
role tagging
→ fallback embeddings
```

### Bowler Role Classification Taxonomy

To handle cold-start embeddings and prevent representation collapse for debutant bowlers, we utilize a deterministic, heuristic-based role classification system. Bowlers are grouped into one of 7 archetypes based on their historical bowling phase ratios. 

When a new bowler enters the simulation with no historical data, they inherit the cluster-average embedding of their assigned role, offset by a small, player-specific deterministic Gaussian noise vector.

| Role Class | Count | Classification Condition (Sequential) | Averages (PP / Mid / Death) | Usage Context |
| :--- | :--- | :--- | :--- | :--- |
| `Unknown_Part_Timer` | 152 | `total_overs <= 9` | 21.0% / 55.5% / 23.5% | Catch-all for extreme low-volume bowlers, part-timers, and debutants lacking enough data for a confident specialized profile. |
| `Middle_Spinner` | 108 | `is_pacer == 0` AND `mid_ratio > 0.6` | 9.4% / 79.2% / 11.4% | Traditional spinners strictly restricted to operating during the middle overs (overs 6-14). |
| `Versatile_Spinner`| 34 | `is_pacer == 0` (Fallback) | 36.2% / 49.4% / 14.4% | Mystery spinners or highly trusted spinners frequently utilized in the Powerplay (e.g., Sunil Narine, R. Ashwin). |
| `Powerplay_Pacer` | 114 | `is_pacer == 1` AND `pp_ratio > 0.41` | 55.0% / 21.2% / 23.8% | Swing and seam specialists structurally utilized at the top of the innings to extract early movement. |
| `Death_Pacer` | 56 | `is_pacer == 1` AND `death_ratio > 0.33` | 25.1% / 35.9% / 39.0% | Yorkers/slower-ball specialists retained heavily for the final 5 overs (e.g., Jasprit Bumrah, Lasith Malinga). |
| `Middle_Pacer` | 45 | `is_pacer == 1` AND `mid_ratio > 0.55` | 11.4% / 69.1% / 19.5% | Enforcers or hit-the-deck pacers utilized primarily when the field is spread in the middle phase. |
| `Versatile_Pacer` | 33 | `is_pacer == 1` (Fallback) | 31.3% / 42.2% / 26.5% | Highly balanced all-phase pacers capable of bowling across Powerplay, Middle, and Death interchangeably. |

> **Note:** The classification logic is strictly sequential. It must evaluate `total_overs` first, then branch by `is_pacer`, and then evaluate phase ratios in the exact order listed above.