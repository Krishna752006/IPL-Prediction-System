Here are the exact formulas being used.

---

## 1. Total appearances

Number of innings where batter appeared.

T=\sum_{i=1}^{11} c_i

Where:

* (c_i) = count of batting at position (i)

Example (Rayudu):

```text
2+15+42+47+44+30+4=184
```

---

## 2. Weighted mean batting position

Average batting position weighted by frequency.

\bar{x}=\frac{\sum_{i=1}^{11} i\cdot c_i}{\sum_{i=1}^{11} c_i}

Where:

* (i) = batting position
* (c_i) = number of times batted there

For Rayudu:

Numerator:

```text
(1×2)
+(2×15)
+(3×42)
+(4×47)
+(5×44)
+(6×30)
+(7×4)
```

```text
= 774
```

Denominator:

```text
184
```

So:

\bar{x}=\frac{774}{184}=4.2065

Interpretation:

```text
lower → bats higher
higher → bats lower
```

---

## 3. Single-position dominance ratio

Measures positional stability at one exact batting slot.

D_{single}=\frac{\max(c_i)}{T}

Where:

* (\max(c_i)) = largest count among positions
* (T) = total appearances

For Rayudu:

Largest count:

```text
47
```

Total:

```text
184
```

So:

D_{single}=\frac{47}{184}=0.2554

Interpretation:

```text
high → specialist
low → floater
```

---

## 4. Bucket count

Counts batting appearances inside a role bucket.

General formula:

B_r=\sum_{i\in r} c_i

Where:

* (r) = role bucket
* (c_i) = position counts

Example buckets:

```text
opener       = {1,2}
top_order    = {1,2,3}
middle_order = {4,5,6}
finisher     = {5,6,7,8}
tail         = {8,9,10,11}
```

Example (Rayudu middle order):

```text
47+44+30
=121
```

So:

B_{middle}=c_4+c_5+c_6

---

## 5. Bucket dominance ratio

Measures evidence for a batting role.

General:

D_r=\frac{B_r}{T}

Where:

* (B_r) = bucket count
* (T) = total appearances

Example (Rayudu middle order):

```text
121 / 184
```

So:

D_{middle}=\frac{121}{184}=0.6576

Interpretation:

```text
higher = stronger evidence
```

| Class Name               | Total Count | Defining Condition (Rule)                                                    | Usage in Pipeline                                                                 |
|--------------------------|-------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Unknown (Low Data)       | 240         | `total_appearances <= 4`                                                        | Excluded. Filters out statistical noise and bowlers who rarely batted.           |
| Pure Opener              | 66          | `opener_dominance >= 0.65`                                                      | Averaged. Base embedding for pure opening debutants.                             |
| Pure Top Order           | 29          | `top_order_dominance >= 0.55 AND opener_dominance < 0.45`                       | Averaged. Base embedding for first-drop / #3 debutants.                          |
| Pure Middle Order        | 83          | `middle_order_dominance >= 0.65 AND top_order_dominance < 0.50`                 | Averaged. Base embedding for pure #4, #5, #6 batters.                            |
| Pure Finisher            | 76          | `finisher_dominance >= 0.65 AND middle_order_dominance < 0.50`                  | Averaged. Base embedding for death-over debutants.                               |
| Pure Tail                | 82          | `tail_dominance >= 0.85 AND finisher_dominance < 0.50`                          | Averaged. Base embedding for pure bowler debutants.                              |
| Versatile Anchor         | 49          | `(Pos 3 + Pos 4 + Pos 5) / total_appearances >= 0.40`                           | Averaged. Base embedding for situational middle-order floaters.                  |
| Excluded / Floater       | 70          | `Fails all constraints above.`                                                   | Excluded. Extreme floaters. Prevents representation pollution in averages.       |