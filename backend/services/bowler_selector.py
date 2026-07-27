"""
Bowler selection.

Split into two clearly separate pieces:

1. `eligible_bowlers()` — the actual cricket rules (max MAX_OVERS_PER_BOWLER
   overs per bowler, no consecutive overs). This is what builds the
   "available bowlers" array.
2. `select_bowler_from_pool()` / POST /select-bowler — the "dummy API" you
   asked for: it takes that array and returns one bowler. For now that's
   just `random.choice`. It has zero knowledge of overs/consecutive-over
   rules — it's a pure "array in, one out" picker, so it's trivial to swap
   for a real ranking/scoring model later without touching the rule logic
   or match_engine.py.

Safety valve: a T20 innings needs at least 5 distinct bowlers (20 overs /
4-over cap). DC's actual 2026 XI has exactly 5 (the mathematical minimum —
zero slack: all 5 must bowl exactly 4 overs each). Picking uniformly at
random among *all* eligible bowlers can paint that zero-slack case into a
corner late in an innings (e.g. 4 bowlers already at their cap, leaving only
the one who also bowled the previous over) — no valid pick exists at that
point no matter what the "dummy" picker does. So `pick_next_bowler` first
narrows the eligible array down to whoever has bowled the *fewest* overs so
far (ties broken by the random picker, same as before) — this keeps overs
spread evenly across the pool as the innings progresses, which prevents
that corner from ever being reached. Verified with 200 DC innings across
random seeds: zero consecutive-over or 4-over-cap violations (see the
verification notes shipped with this change). The 4-over cap itself is
still never relaxed; the "relax no-consecutive" fallback below is now a
true last resort that shouldn't fire for any 2026 squad.
"""
from __future__ import annotations

import random

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ml_config import MAX_OVERS_PER_BOWLER
from services.squads import Player


class BowlerUsage(BaseModel):
    overs_bowled: float = 0.0  # legal balls / 6
    last_over_bowled: int | None = None  # over number (0-indexed), or None


def eligible_bowlers(
    candidates: list[Player],
    usage: dict[str, BowlerUsage],
    current_over: int,
    relax_consecutive_rule: bool = False,
) -> list[Player]:
    """Applies the actual cricket rules and returns the array of bowlers
    allowed to bowl `current_over`."""
    out = []
    for p in candidates:
        u = usage.get(p.name, BowlerUsage())
        if u.overs_bowled >= MAX_OVERS_PER_BOWLER:
            continue
        if not relax_consecutive_rule and u.last_over_bowled == current_over - 1:
            continue
        out.append(p)
    return out


def select_bowler_from_pool(
    available_bowlers: list[Player], rng: random.Random | None = None
) -> Player:
    """The "dummy API": given an array of already-eligible bowlers, return
    one. Random for now — swap this one line for a real model later."""
    rng = rng or random
    return rng.choice(available_bowlers)


def pick_next_bowler(
    candidates: list[Player],
    usage: dict[str, BowlerUsage],
    current_over: int,
    rng: random.Random | None = None,
) -> Player:
    """Rules -> array -> pick, with load-balancing so zero-slack squads
    (see module docstring) never get forced into a rule violation."""
    pool = eligible_bowlers(candidates, usage, current_over)
    if not pool:
        # Safety valve documented above — true last resort now.
        pool = eligible_bowlers(
            candidates, usage, current_over, relax_consecutive_rule=True
        )
    if not pool:
        raise RuntimeError(
            "No eligible bowler available — squad has fewer bowlers than a "
            "20-over innings requires (need >=5 under the 4-over cap)."
        )

    least_overs = min(usage.get(p.name, BowlerUsage()).overs_bowled for p in pool)
    pool = [p for p in pool if usage.get(p.name, BowlerUsage()).overs_bowled == least_overs]

    return select_bowler_from_pool(pool, rng=rng)


# --------------------------------------------------------------------------
# Standalone router — literally "another API": array of available bowlers
# in, one bowler out. No rule knowledge here on purpose (see module docstring).
# --------------------------------------------------------------------------

router = APIRouter(prefix="/select-bowler", tags=["simulation"])


class SelectBowlerRequest(BaseModel):
    available_bowlers: list[str] = Field(
        ..., min_length=1, description="Names of bowlers eligible to bowl this over"
    )


class SelectBowlerResponse(BaseModel):
    bowler: str


@router.post("", response_model=SelectBowlerResponse)
def select_bowler(req: SelectBowlerRequest) -> SelectBowlerResponse:
    chosen = random.choice(req.available_bowlers)
    return SelectBowlerResponse(bowler=chosen)
