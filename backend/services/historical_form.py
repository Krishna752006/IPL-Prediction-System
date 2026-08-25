"""
Recent-form stats for the "last 3 matches" model inputs — sourced from your
real recent_form_batter.parquet / recent_form_bowler.parquet
(generate_offline_stats.py's output). If a player has no record there
(a 2026 debutant), every field is 0 — which is also just the cricket-correct
answer for someone who's never played a match.

IMPORTANT data-shape note: generate_offline_stats.py only writes out
`matchId, batsman, history_matches, last_1_runs, last_1_balls, last_2_*,
last_3_*` (see `batter_cols`/`bowler_cols`) — it does NOT save that row's
own actual runs/balls for that match, only the *shifted* history going into
it. So the most recent saved row for a player already has some lag built
in: its `last_1_*` is that player's form from the match immediately before
their most recent recorded one, not from their true latest game. Using
that row's last_1/2/3 fields directly as the "recent form entering the
2026 match" is the closest honest read of the data — reconstructing
anything finer isn't possible without a runs/balls column that just isn't
in this file. If a player has fewer than 3 prior matches, the missing
last_i fields are already 0 in the source parquet (generate_offline_stats.py
fills shifted NaNs with 0) — read through as-is, not overridden.

`history_matches` is a cumcount (0-indexed matches *before* this row), so a
player's row with the highest history_matches value is their most recent
recorded match — no `date` column survives into this file, so that's how
"most recent" is determined (idxmax, not a date sort).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from ml_config import RECENT_FORM_BATTER_PATH, RECENT_FORM_BOWLER_PATH

logger = logging.getLogger(__name__)

_ZERO3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BatterForm:
    history_matches: int = 0
    last_runs: tuple[float, float, float] = _ZERO3
    last_balls: tuple[float, float, float] = _ZERO3


@dataclass(frozen=True)
class BowlerForm:
    history_matches: int = 0
    last_runs_conceded: tuple[float, float, float] = _ZERO3
    last_balls_bowled: tuple[float, float, float] = _ZERO3


@lru_cache(maxsize=1)
def _load_batter_table() -> dict[str, BatterForm]:
    if not os.path.exists(RECENT_FORM_BATTER_PATH):
        logger.warning(
            "recent_form_batter.parquet missing at %s — every batter's "
            "recent form will default to 0.",
            RECENT_FORM_BATTER_PATH,
        )
        return {}
    df = pd.read_parquet(RECENT_FORM_BATTER_PATH)
    latest = df.loc[df.groupby("batsman")["history_matches"].idxmax()]
    return {
        row["batsman"]: BatterForm(
            history_matches=int(row["history_matches"]) + 1,
            last_runs=(row["last_1_runs"], row["last_2_runs"], row["last_3_runs"]),
            last_balls=(row["last_1_balls"], row["last_2_balls"], row["last_3_balls"]),
        )
        for _, row in latest.iterrows()
    }


@lru_cache(maxsize=1)
def _load_bowler_table() -> dict[str, BowlerForm]:
    if not os.path.exists(RECENT_FORM_BOWLER_PATH):
        logger.warning(
            "recent_form_bowler.parquet missing at %s — every bowler's "
            "recent form will default to 0.",
            RECENT_FORM_BOWLER_PATH,
        )
        return {}
    df = pd.read_parquet(RECENT_FORM_BOWLER_PATH)
    latest = df.loc[df.groupby("bowler")["history_matches"].idxmax()]
    return {
        row["bowler"]: BowlerForm(
            history_matches=int(row["history_matches"]) + 1,
            last_runs_conceded=(
                row["last_1_runs_conceded"],
                row["last_2_runs_conceded"],
                row["last_3_runs_conceded"],
            ),
            last_balls_bowled=(
                row["last_1_balls_bowled"],
                row["last_2_balls_bowled"],
                row["last_3_balls_bowled"],
            ),
        )
        for _, row in latest.iterrows()
    }


def get_batter_form(player_name: str) -> BatterForm:
    return _load_batter_table().get(player_name, BatterForm())


def get_bowler_form(player_name: str) -> BowlerForm:
    return _load_bowler_table().get(player_name, BowlerForm())
