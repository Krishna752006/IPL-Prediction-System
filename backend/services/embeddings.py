"""
Loads ml-service/saved_seasons/static_embeddings_<EMBEDDINGS_SEASON>.json and
resolves per-ball embedding vectors.

Three lookup tiers, in order (per your instructions — "do not return
zeros" for unknown/debutant players):

  1. Exact match — the player's own vector from the embeddings file.
  2. Class average — for a player missing from the embeddings file (a 2026
     debutant), look up their role in batter_classification.json /
     bowler_classification.json (PLAYER_TO_BATTER_ROLE /
     PLAYER_TO_BOWLER_ROLE, built exactly like your snippet), then average
     the embeddings of every *known* player who shares that role, plus
     Gaussian noise (std=EMBEDDING_NOISE_STD).
  3. Global average — if the player isn't even in the classification files,
     average every known player's vector instead of guessing a role, plus
     the same noise. Weaker than tier 2, but still not a zero vector.

Tiers 2 and 3 are randomised, so they're resolved once per player per match
(cached) via EmbeddingResolver, not re-sampled every ball — a debutant's
"typical embedding for their role" shouldn't drift ball to ball.

The season embedding is handled separately per your instruction: average
the last SEASON_AVG_LAST_N (5) seasons' embeddings + the same noise,
resolved once per match (there's no "2026" entry to look up anyway).

UPDATE (train_embeddings.py merge): the trained model used to be a Keras
LSTM that consumed a single flat per-ball feature vector — the 32 raw
numeric columns plus these embedding vectors, pre-concatenated by
feature_builder.py before ever reaching the model (see the old
`build_ball_vector`). The new TabTransformerLSTM (trained by
train_embeddings.py, PyTorch) does its OWN nn.Embedding lookup from
integer indices *inside* the model — `save_static_embeddings_to_json`
reads `model.batter_embedding.weight` etc. directly, confirming these are
real embedding submodules, not precomputed inputs.

That's a problem for exactly the players/venues this file exists to
handle: train_embeddings.py never exports `player2idx`/`venue2idx`, so we
have no way to recover which row a given known player actually occupies
in the model's trained embedding table — and even if we did, a 2026
debutant or an averaged "last 5 seasons" vector was never any single
trained row to begin with. So instead of trying to feed the model real
indices, `MatchEmbeddingContext`/`build_match_context` below build a
small, disposable, match-local embedding table (containing just this
match's ~24 players + 1 venue), resolved with the exact same tiered
exact/class-average/global-average(+noise) logic as everything else in
this file, and `model_runner.TrainedModelRunner.load_match_context` swaps
the model's embedding submodules for these tables for the duration of the
match. See that function's docstring for the concurrency caveat this
introduces. If you later add a `player2idx`/`venue2idx` export to
train_embeddings.py, this indirection could shrink to "index known
players/venues directly, only build a local table for the fallback
tiers" — but the debutant/venue/season averaging would still need some
version of this trick, since blended vectors aren't real trained rows.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ml_config import (
    BATTER_CLASSIFICATION_PATH,
    BATTER_EMB_DIM,
    BOWLER_CLASSIFICATION_PATH,
    BOWLER_EMB_DIM,
    EMBEDDING_NOISE_STD,
    EMBEDDINGS_DIR,
    EMBEDDINGS_SEASON,
    MATCH_STATE_EMB_DIM,
    NON_STRIKER_EMB_DIM,
    SEASON_AVG_LAST_N,
    SEASON_EMB_DIM,
    VENUE_EMB_DIM,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load() -> dict:
    path = os.path.join(EMBEDDINGS_DIR, f"static_embeddings_{EMBEDDINGS_SEASON}.json")
    if not os.path.exists(path):
        logger.warning(
            "Embeddings file missing at %s — every embedding lookup will "
            "fall back to zero vectors.",
            path,
        )
        return {"players": {}, "venues": {}, "season": {}, "match_state": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_classification_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (PLAYER_TO_BATTER_ROLE, PLAYER_TO_BOWLER_ROLE), exactly as in
    your snippet — just tolerant of a missing file."""

    def _load_one(path: str, label: str) -> dict[str, list[str]]:
        if not os.path.exists(path):
            logger.warning(
                "%s classification file missing at %s — class-average "
                "embedding fallback disabled for %s, will use the global "
                "average instead.",
                label,
                path,
                label,
            )
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    batter_classes = _load_one(BATTER_CLASSIFICATION_PATH, "batter")
    bowler_classes = _load_one(BOWLER_CLASSIFICATION_PATH, "bowler")

    player_to_batter_role = {
        p: role for role, players in batter_classes.items() for p in players
    }
    player_to_bowler_role = {
        p: role for role, players in bowler_classes.items() for p in players
    }
    return player_to_batter_role, player_to_bowler_role


def _get_exact(field: str, key, dim: int) -> np.ndarray | None:
    players = _load().get("players", {})
    entry = players.get(str(key)) if key is not None else None
    if not entry or field not in entry:
        return None
    return np.asarray(entry[field], dtype=np.float32)


@lru_cache(maxsize=None)
def _role_average(role_map_kind: str, role: str, field: str) -> np.ndarray | None:
    """Deterministic (no noise) average of `field` across every known
    player sharing `role`. role_map_kind is 'batter' or 'bowler'."""
    player_to_batter_role, player_to_bowler_role = _load_classification_maps()
    player_to_role = (
        player_to_batter_role if role_map_kind == "batter" else player_to_bowler_role
    )

    players = _load().get("players", {})
    vectors = [
        np.asarray(entry[field], dtype=np.float32)
        for name, entry in players.items()
        if player_to_role.get(name) == role and field in entry
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


@lru_cache(maxsize=None)
def _global_average(field: str) -> np.ndarray | None:
    players = _load().get("players", {})
    vectors = [
        np.asarray(entry[field], dtype=np.float32)
        for entry in players.values()
        if field in entry
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _resolve_static(
    player_name: str, field: str, dim: int, role_map_kind: str
) -> tuple[np.ndarray, str]:
    """Deterministic part of resolution (no noise). Returns (vector, tier)
    where tier is 'exact' | 'class_average' | 'global_average' | 'zero'."""
    exact = _get_exact(field, player_name, dim)
    if exact is not None:
        return exact, "exact"

    player_to_batter_role, player_to_bowler_role = _load_classification_maps()
    role_map = (
        player_to_batter_role if role_map_kind == "batter" else player_to_bowler_role
    )
    role = role_map.get(player_name)
    if role is not None:
        avg = _role_average(role_map_kind, role, field)
        if avg is not None:
            return avg, "class_average"

    global_avg = _global_average(field)
    if global_avg is not None:
        return global_avg, "global_average"

    logger.warning(
        "No embedding, classification, or global average available for "
        "'%s' (%s) — falling back to a zero vector.",
        player_name,
        field,
    )
    return np.zeros(dim, dtype=np.float32), "zero"


@lru_cache(maxsize=1)
def _season_base_vector() -> np.ndarray:
    """Deterministic (no noise) average of the last SEASON_AVG_LAST_N
    seasons' embeddings."""
    season_map = _load().get("season", {})
    if not season_map:
        return np.zeros(SEASON_EMB_DIM, dtype=np.float32)
    years = sorted((int(y) for y in season_map.keys()), reverse=True)
    last_n = years[:SEASON_AVG_LAST_N]
    vectors = [np.asarray(season_map[str(y)], dtype=np.float32) for y in last_n]
    return np.mean(vectors, axis=0)


def venue_vec(venue_name: str) -> np.ndarray:
    venues = _load().get("venues", {})
    vec = venues.get(venue_name)
    if vec is None:
        return np.zeros(VENUE_EMB_DIM, dtype=np.float32)
    return np.asarray(vec, dtype=np.float32)


def match_state_vec(state_id: int = 0) -> np.ndarray:
    """
    The real semantics of `match_state_id` (which of the 6 clustered match
    states a ball belongs to) live in the ml-service feature-engineering
    code, which I don't have. Defaulting to a neutral index (0) for every
    ball is a documented placeholder — swap in the real bucketing logic here
    if you have it.
    """
    states = _load().get("match_state", [])
    if not states or state_id >= len(states):
        return np.zeros(MATCH_STATE_EMB_DIM, dtype=np.float32)
    return np.asarray(states[state_id], dtype=np.float32)


def list_venues() -> list[str]:
    return list(_load().get("venues", {}).keys())


class EmbeddingResolver:
    """
    Per-match resolver: adds noise on top of the tier-2/3 fallbacks and the
    season average, caching so each player (and the season vector) gets
    exactly one noise draw per match rather than a fresh one every ball.
    Exact-match (tier 1) lookups are never noised.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._season_vec: np.ndarray | None = None

    def _resolve(
        self, player_name: str, field: str, dim: int, role_map_kind: str
    ) -> np.ndarray:
        cache_key = (player_name, field)
        if cache_key in self._cache:
            return self._cache[cache_key]

        vector, tier = _resolve_static(player_name, field, dim, role_map_kind)
        if tier in ("class_average", "global_average"):
            vector = vector + self._rng.normal(
                0.0, EMBEDDING_NOISE_STD, size=dim
            ).astype(np.float32)
        self._cache[cache_key] = vector
        return vector

    def batter_vec(self, player_name: str) -> np.ndarray:
        return self._resolve(player_name, "batter_embedding", BATTER_EMB_DIM, "batter")

    def non_striker_vec(self, player_name: str) -> np.ndarray:
        return self._resolve(
            player_name, "non_striker_embedding", NON_STRIKER_EMB_DIM, "batter"
        )

    def bowler_vec(self, player_name: str) -> np.ndarray:
        return self._resolve(player_name, "bowler_embedding", BOWLER_EMB_DIM, "bowler")

    def season_vec(self) -> np.ndarray:
        if self._season_vec is None:
            base = _season_base_vector()
            self._season_vec = base + self._rng.normal(
                0.0, EMBEDDING_NOISE_STD, size=SEASON_EMB_DIM
            ).astype(np.float32)
        return self._season_vec


@dataclass
class MatchEmbeddingContext:
    """
    Match-local embedding tables + index maps, built once per match by
    `build_match_context` (see the module docstring's "UPDATE" section for
    why this exists). Row 0 in every matrix is a reserved all-zero "pad"
    row — used only for padding a short innings' sequence up to SEQ_LEN in
    model_runner._pad_categorical, never a real player/venue/season — so
    every real entry's index starts at 1.

    `batter_index` doubles as the index map for `non_striker_matrix` too:
    a player's "batter" and "non-striker" embeddings are different vectors
    (different fields in the static JSON), but the same player always gets
    the same *row number* in both matrices, so one dict covers both.
    """

    batter_index: dict[str, int]
    batter_matrix: np.ndarray  # (len(batter_index) + 1, BATTER_EMB_DIM)
    non_striker_matrix: np.ndarray  # same shape, indexed via batter_index
    bowler_index: dict[str, int]
    bowler_matrix: np.ndarray  # (len(bowler_index) + 1, BOWLER_EMB_DIM)
    venue_matrix: np.ndarray  # (2, VENUE_EMB_DIM): [pad, this match's venue]
    season_matrix: np.ndarray  # (2, SEASON_EMB_DIM): [pad, last-N-season average]

    def player_idx(self, name: str) -> int:
        """Index into batter_matrix / non_striker_matrix for `name`. Falls
        back to the pad row (0) if `name` somehow isn't in this match's
        squads — shouldn't happen if build_match_context was given the
        full playing squads, but keeps this from raising mid-simulation."""
        return self.batter_index.get(name, 0)

    def bowler_idx(self, name: str) -> int:
        return self.bowler_index.get(name, 0)

    @property
    def venue_idx(self) -> int:
        return 1  # row 0 is pad; every match has exactly one venue at row 1

    @property
    def season_idx(self) -> int:
        return 1  # row 0 is pad; every match has exactly one season vector


def build_match_context(
    batting_pool_names: list[str],
    bowling_pool_names: list[str],
    venue_name: str,
    resolver: EmbeddingResolver,
) -> MatchEmbeddingContext:
    """
    batting_pool_names: every player who could ever be a striker/non-striker
    this match (both teams' full batting orders — 11 + 11).
    bowling_pool_names: every player who could ever be given the ball
    (both teams' bowling_pool — designated bowlers + the bowling-only 12th
    man).
    """
    batters = list(dict.fromkeys(batting_pool_names))  # de-dupe, keep order
    batter_index = {name: i + 1 for i, name in enumerate(batters)}
    batter_matrix = np.zeros((len(batters) + 1, BATTER_EMB_DIM), dtype=np.float32)
    non_striker_matrix = np.zeros(
        (len(batters) + 1, NON_STRIKER_EMB_DIM), dtype=np.float32
    )
    for name, idx in batter_index.items():
        batter_matrix[idx] = resolver.batter_vec(name)
        non_striker_matrix[idx] = resolver.non_striker_vec(name)

    bowlers = list(dict.fromkeys(bowling_pool_names))
    bowler_index = {name: i + 1 for i, name in enumerate(bowlers)}
    bowler_matrix = np.zeros((len(bowlers) + 1, BOWLER_EMB_DIM), dtype=np.float32)
    for name, idx in bowler_index.items():
        bowler_matrix[idx] = resolver.bowler_vec(name)

    venue_matrix = np.zeros((2, VENUE_EMB_DIM), dtype=np.float32)
    venue_matrix[1] = venue_vec(venue_name)

    season_matrix = np.zeros((2, SEASON_EMB_DIM), dtype=np.float32)
    season_matrix[1] = resolver.season_vec()

    return MatchEmbeddingContext(
        batter_index=batter_index,
        batter_matrix=batter_matrix,
        non_striker_matrix=non_striker_matrix,
        bowler_index=bowler_index,
        bowler_matrix=bowler_matrix,
        venue_matrix=venue_matrix,
        season_matrix=season_matrix,
    )
