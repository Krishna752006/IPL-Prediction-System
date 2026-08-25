"""
Loads the venue_phase_avg.json produced by generate_offline_stats.py and
exposes `venue_phase_avg(venue, phase, season)`, mirroring the
(season_bucket, venue, phase) -> scoring_index lookup built there.

generate_offline_stats.py buckets season into
  [2007,2013) -> "2008_2013", [2013,2019) -> "2014_2019",
  [2019,2022) -> "2020_2022", [2022,2025] -> "2023_2025"
2026 falls outside every bin, so it's clamped to the latest bucket
("2023_2025") — the same clamp implicitly applies whenever this file
hasn't been regenerated for the current season yet.

If the file doesn't exist (it isn't part of what you've given me), every
lookup returns the neutral index 1.0 (== "scores at the league-average
rate for this phase"), with a single startup warning instead of crashing.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from functools import lru_cache

from ml_config import VENUE_PHASE_AVG_PATH

logger = logging.getLogger(__name__)

_SEASON_BUCKET_BINS = [2007, 2013, 2019, 2022, 2025]
_SEASON_BUCKET_LABELS = ["2008_2013", "2014_2019", "2020_2022", "2023_2025"]

NEUTRAL_SCORING_INDEX = 1.0


def _season_bucket(season: int) -> str:
    if season > _SEASON_BUCKET_BINS[-1]:
        return _SEASON_BUCKET_LABELS[-1]  # clamp future seasons (e.g. 2026)
    for lo, hi, label in zip(
        _SEASON_BUCKET_BINS, _SEASON_BUCKET_BINS[1:], _SEASON_BUCKET_LABELS
    ):
        if lo < season <= hi:
            return label
    return _SEASON_BUCKET_LABELS[0]


@lru_cache(maxsize=1)
def _load() -> dict:
    if not os.path.exists(VENUE_PHASE_AVG_PATH):
        logger.warning(
            "venue_phase_avg.json missing at %s — venue_phase_avg feature "
            "will default to the neutral index (%.1f) for every ball.",
            VENUE_PHASE_AVG_PATH,
            NEUTRAL_SCORING_INDEX,
        )
        return {}
    with open(VENUE_PHASE_AVG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # keys are stringified tuples: "('2023_2025', 'Venue', 'PP')"
    parsed = {}
    for k, v in raw.items():
        try:
            key_tuple = ast.literal_eval(k)
        except (ValueError, SyntaxError):
            continue
        parsed[key_tuple] = v
    return parsed


def venue_phase_avg(venue: str, phase: str, season: int) -> float:
    table = _load()
    if not table:
        return NEUTRAL_SCORING_INDEX
    bucket = _season_bucket(season)
    entry = table.get((bucket, venue, phase))
    if entry is None:
        return NEUTRAL_SCORING_INDEX
    return float(entry.get("scoring_index", NEUTRAL_SCORING_INDEX))
