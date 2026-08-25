"""
Builds the two per-timestep tensors the new TabTransformerLSTM (trained by
train_embeddings.py) expects:

    numerical:   the 32 raw feature_columns, unchanged from before.
    categorical: 6 integer indices, in this order —
        [batter_idx, non_striker_idx, bowler_idx, venue_idx, season_idx,
         match_state_idx]
        matching train_embeddings.py's own usage of the categorical tensor
        (`categorical[:, -1, 4]` for season — hence season sitting at
        position 4 here, with match_state inferred as the trailing 6th
        column since save_static_embeddings_to_json exports exactly six
        embedding tables: batter, non_striker, bowler, venue, season,
        match_state).

UPDATE (train_embeddings.py merge): this used to also concatenate 226 dims
of precomputed embedding vectors onto the 32 numeric ones (258 total),
because the old Keras model took one flat vector per ball. The new model
embeds categorical indices itself, so those indices — not vectors — are
what this file now produces. The indices themselves aren't the model's
*real* trained row numbers (see services/embeddings.py's module docstring
"UPDATE" section for why not, and MatchEmbeddingContext for the fix);
they're row numbers into a match-local table that model_runner swaps in
for the actual match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ml_config import RAW_FEATURE_COLUMNS
from services import embeddings, offline_stats
from services.match_state import InningsState


def _phase_for_over(over_idx: int) -> str:
    # matches generate_offline_stats.py: pd.cut(over, [-1,5.9,14.9,20])
    if over_idx <= 5:
        return "PP"
    if over_idx <= 14:
        return "Middle"
    return "Death"


def _rr_momentum(innings: InningsState, window: int = 12) -> float:
    """Recent run-rate (last `window` legal balls) minus the innings'
    overall run-rate so far. Documented assumption — the exact definition
    used at training time isn't available to me; this is a reasonable
    momentum signal consistent with the column name."""
    if innings.legal_balls == 0:
        return 0.0
    overall_rr = innings.score / (innings.legal_balls / 6.0)
    recent = innings.recent_ball_runs[-window:]
    if not recent:
        return 0.0
    recent_rr = sum(recent) / (len(recent) / 6.0)
    return recent_rr - overall_rr


@dataclass
class BallContext:
    inning_no: int
    over_idx: int  # 0-indexed
    ball_in_over: int  # 1..6 (legal-ball position)
    legal_balls_before: int
    wickets_before: int
    target: int | None
    striker_name: str
    non_striker_name: str
    bowler_name: str
    bowler_is_pacer: bool
    venue: str
    toss_winner_is_batting_team: bool
    batter_history_matches: int
    batter_last_runs: tuple[float, float, float]
    batter_last_balls: tuple[float, float, float]
    bowler_history_matches: int
    bowler_last_runs_conceded: tuple[float, float, float]
    bowler_last_balls_bowled: tuple[float, float, float]
    rr_momentum: float
    season: int


def build_raw_features(ctx: BallContext) -> dict[str, float]:
    phase = _phase_for_over(ctx.over_idx)
    balls_remaining = max(0, 120 - ctx.legal_balls_before)
    current_run_rate = 0.0
    if ctx.legal_balls_before > 0:
        # caller fills this in via innings.score before calling; kept out of
        # BallContext to avoid duplicating InningsState — see build_ball_features
        pass

    theta = 2 * math.pi * ctx.ball_in_over / 6.0

    row = {
        "inning": float(ctx.inning_no),
        "over": float(ctx.over_idx),
        "total_balls": float(ctx.legal_balls_before),
        "balls_remaining": float(balls_remaining),
        "phase_pp": 1.0 if phase == "PP" else 0.0,
        "phase_middle": 1.0 if phase == "Middle" else 0.0,
        "phase_death": 1.0 if phase == "Death" else 0.0,
        "target": float(ctx.target or 0),
        "is_pacer": 1.0 if ctx.bowler_is_pacer else 0.0,
        "wickets_before": float(ctx.wickets_before),
        "sin_ball": math.sin(theta),
        "cos_ball": math.cos(theta),
        "rr_momentum": float(ctx.rr_momentum),
        "toss_won": 1.0 if ctx.toss_winner_is_batting_team else 0.0,
        "venue_phase_avg": offline_stats.venue_phase_avg(ctx.venue, phase, ctx.season),
        "batter_history_matches": float(ctx.batter_history_matches),
        "last_1_runs": float(ctx.batter_last_runs[0]),
        "last_1_balls": float(ctx.batter_last_balls[0]),
        "last_2_runs": float(ctx.batter_last_runs[1]),
        "last_2_balls": float(ctx.batter_last_balls[1]),
        "last_3_runs": float(ctx.batter_last_runs[2]),
        "last_3_balls": float(ctx.batter_last_balls[2]),
        "bowler_history_matches": float(ctx.bowler_history_matches),
        "last_1_runs_conceded": float(ctx.bowler_last_runs_conceded[0]),
        "last_1_balls_bowled": float(ctx.bowler_last_balls_bowled[0]),
        "last_2_runs_conceded": float(ctx.bowler_last_runs_conceded[1]),
        "last_2_balls_bowled": float(ctx.bowler_last_balls_bowled[1]),
        "last_3_runs_conceded": float(ctx.bowler_last_runs_conceded[2]),
        "last_3_balls_bowled": float(ctx.bowler_last_balls_bowled[2]),
        # filled by caller (needs live score/target which BallContext
        # doesn't carry to keep this pure):
        "current_run_rate": 0.0,
        "percentage_target_achieved": 0.0,
        "required_run_rate": 0.0,
    }
    return row


def build_ball_features(
    ctx: BallContext,
    current_run_rate: float,
    percentage_target_achieved: float,
    required_run_rate: float,
    match_ctx: "embeddings.MatchEmbeddingContext",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (numerical_row, categorical_row) for one ball — see the
    module docstring for what each contains."""
    row = build_raw_features(ctx)
    row["current_run_rate"] = current_run_rate
    row["percentage_target_achieved"] = percentage_target_achieved
    row["required_run_rate"] = required_run_rate

    numeric = np.array([row[c] for c in RAW_FEATURE_COLUMNS], dtype=np.float32)

    categorical = np.array(
        [
            match_ctx.player_idx(ctx.striker_name),
            match_ctx.player_idx(ctx.non_striker_name),
            match_ctx.bowler_idx(ctx.bowler_name),
            match_ctx.venue_idx,
            match_ctx.season_idx,
            0,  # match_state: CONFIRMED (not guessed) via tabtransformer_lstm.py —
            # match_state_embedding is declared with padding_idx=0, so index 0
            # isn't some trained "neutral" state, it's the model's frozen,
            # permanently-zero pad row. Feeding 0 for every ball means this
            # feature contributes exactly zero signal to every prediction
            # right now — not a wrong-but-plausible guess, just an absent
            # one. The real per-ball state-clustering logic (whatever feeds
            # match_state_id during training) lives in the ml-service data
            # pipeline, which I don't have.
        ],
        dtype=np.int64,
    )
    return numeric, categorical
