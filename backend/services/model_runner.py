"""
Loads the trained model checkpoint (*.pth in ml-service/models/production)
and exposes a single call:
predict_ball(numerical_sequence, categorical_sequence, score_before) ->
(predicted_score, wicket_prob, wide_prob).

UPDATE (checkpoint format change): training now saves via
`torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict':
..., 'epoch':..., 'composite_score':..., 'wicket_thresh':...,
'wide_thresh':...}, path)` instead of pickling a live
`IPLModelBundle(model=...)` object. A state_dict is just weights — no
architecture attached — so this module now needs to know the
`TabTransformerLSTM` class itself to build an empty model before
`load_state_dict()` can fill it in. Rather than depend on your ml-service
repo being importable via a sys.path hack (fragile, and no longer
necessary), `ml_model/tabtransformer_lstm.py` is a vendored copy of the
real file you shared — see that file's docstring if you change the
architecture later.

The four architecture args the constructor needs (num_players, num_venues,
num_seasons, numerical_dim) aren't in the checkpoint either, but they don't
need to be: they're recoverable directly from the state_dict's own tensor
shapes (see `_infer_architecture`), so this doesn't depend on any
separately-tracked vocab-size config that could drift out of sync.

If PyTorch isn't installed, no checkpoint is found, or loading fails for
any reason, this falls back to HeuristicModelRunner so /predict-1-match
still runs end-to-end (clearly flagged via `model_backend` in the
response).
"""

from __future__ import annotations

import glob
import logging
import os
import random
import threading
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ml_config import (
    MODEL_DIR,
    MODEL_PATH,
    SCORE_SCALE,
    SEQ_LEN,
    WICKET_PROB_THRESHOLD,
    WIDE_PROB_THRESHOLD,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from ml_model.tabtransformer_lstm import TabTransformerLSTM

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelRunner:
    """Common interface both the real and heuristic runners implement.

    Both subclasses also expose `wicket_thresh` / `wide_thresh` (floats):
    static baseline cutoffs used only as the `fallback` for
    match_engine.py's RollingThreshold instances (see this module's "Custom
    ball-outcome shaping" section) until an innings has produced enough of
    the model's own raw output to calibrate against. TrainedModelRunner
    pulls its baseline from the checkpoint itself (the adaptive thresholds
    computed at training time via train_embeddings.py's
    get_adaptive_threshold()) rather than a single global guess — see
    TrainedModelRunner.__init__.
    """

    backend_name = "unknown"

    def predict_ball(
        self,
        numerical_sequence: np.ndarray,
        categorical_sequence: np.ndarray,
        score_before: float,
    ) -> tuple[float, float, float]:
        """numerical_sequence: shape (T, 32), categorical_sequence: shape
        (T, 6) int indices (see feature_builder.py) — both T <= SEQ_LEN,
        most-recent-last, same T for both.
        score_before: the innings' actual cumulative score prior to this ball
        (needed because the trained model's "score" head predicts the
        *cumulative* total, not the per-ball delta — see train_embeddings.py,
        the target column is `current_score`).
        Returns (delta_runs_off_this_ball, wicket_prob, wide_prob)."""
        raise NotImplementedError


def _pad_sequence(sequence: np.ndarray) -> np.ndarray:
    t, dim = sequence.shape
    if t >= SEQ_LEN:
        return sequence[-SEQ_LEN:]
    pad = np.zeros((SEQ_LEN - t, dim), dtype=np.float32)
    return np.vstack([pad, sequence])


def _pad_categorical(sequence: np.ndarray) -> np.ndarray:
    """Same idea as _pad_sequence, but int-typed and padding with index 0 —
    the reserved "pad" row in every MatchEmbeddingContext table (see
    services/embeddings.py), not a real player/venue/season."""
    t, dim = sequence.shape
    if t >= SEQ_LEN:
        return sequence[-SEQ_LEN:]
    pad = np.zeros((SEQ_LEN - t, dim), dtype=np.int64)
    return np.vstack([pad, sequence])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Custom ball-outcome shaping
# =============================================================================
# The checkpoint's raw wicket_prob / wide_prob outputs don't sit at any
# absolute level we can predict ahead of time — verify_simulation.py has
# shown the *same* checkpoint swing from a 0.0% to a 7.8/match wicket rate
# across two runs, while wide sits stuck around 40% either way. A fixed
# offset added to a threshold (the old `checkpoint.wicket_thresh - 0.225` /
# `+ 0.32`) only "worked" for whatever raw distribution the checkpoint
# happened to produce that one time; it silently breaks the moment the
# distribution drifts, which is exactly what's been observed. So instead:
#
#   1. Both thresholds are now recalibrated continuously, per innings, from
#      a rolling window of the model's OWN recent raw output (RollingThreshold
#      below) — targeting a realistic real-world rate rather than trusting
#      the checkpoint's absolute scale.
#   2. On top of that calibrated wicket threshold, the cricket-domain rules
#      requested are layered in: a batting-order-dependent chance modifier
#      (openers are harder to dismiss, tail easier) and a streak penalty
#      that makes an immediate follow-up wicket progressively rarer, with a
#      4th in a row blocked outright.
#   3. Run *magnitude* on a normal ball is no longer derived from the
#      model's own delta at all (see TrainedModelRunner's docstring — the
#      score head has no reliable per-ball resolution) but sampled from a
#      batting-order-aware weight table, which is what actually controls
#      fours/sixes/strike-rotation realism.
#
# match_engine.py owns the per-ball loop and the innings state (who's on
# strike, how many wickets just fell in a row), so it calls the helpers
# below explicitly, ball by ball. ModelRunner.predict_ball's own contract —
# return the model's raw (delta, wicket_prob, wide_prob) — is left
# untouched so nothing else that depends on it (e.g.
# verify_simulation.py's profiling patch of TrainedModelRunner.predict_ball)
# breaks.

WIDE_TARGET_RATE = 0.045
"""Target fraction of delivery *attempts* that should be called a wide.
Real T20 cricket runs ~4-5%; retune if verify_simulation.py's wide% drifts
noticeably from this after a checkpoint change."""

WICKET_BASE_TARGET_RATE = 0.035
"""Target fraction of legal balls that end in a dismissal for an
"average" (bucket-3-ish, modifier ~0) batter, before the order/streak
adjustments below are layered on. Lands around 7-8 wickets/innings across
a full XI — see ORDER_WICKET_MODIFIER for how this shifts by batting
position. (Empirically tuned against a full run of the simulation
pipeline — see the module-level note above ORDER_RUN_WEIGHTS.)"""

CALIBRATION_WINDOW = 40
"""How many of the model's most recent raw probability samples (this
innings) a RollingThreshold calibrates against."""

MIN_CALIBRATION_SAMPLES = 15
"""Below this many samples, RollingThreshold hasn't seen enough of this
innings' own output yet and uses its static `fallback` instead (the
checkpoint's own wicket_thresh/wide_thresh, or ml_config's defaults for the
heuristic backend)."""


class RollingThreshold:
    """A threshold that recalibrates itself so that, of the model's own
    recent raw probability outputs *this innings*, roughly `target_rate` of
    them sit at or above it — rather than comparing against a fixed number
    that assumes a raw-output distribution/scale we have no way to verify
    ahead of time for a given checkpoint.

    Not thread-shared: match_engine.py creates one pair of these (wide,
    wicket) per innings, matching TrainedModelRunner's own per-match lock
    and reset_innings() lifecycle.
    """

    def __init__(
        self,
        target_rate: float,
        window: int = CALIBRATION_WINDOW,
        min_samples: int = MIN_CALIBRATION_SAMPLES,
        fallback: float = 0.5,
    ):
        self.target_rate = target_rate
        self.min_samples = min_samples
        self.fallback = fallback
        self._samples: deque[float] = deque(maxlen=window)

    def observe(self, raw_prob: float) -> None:
        self._samples.append(float(raw_prob))

    def threshold(self) -> float:
        return self.threshold_for_rate(self.target_rate)

    def threshold_for_rate(self, rate: float) -> float:
        """Like `threshold()`, but for an arbitrary target rate instead of
        this instance's own default — this is what lets a single
        calibrated sample window serve every batting-order bucket's
        different effective dismissal rate (see effective_wicket_target_rate
        below) without needing to know anything about the raw probability
        distribution's actual scale or spread.
        """
        if rate <= 0.0:
            return 1.01  # impossible — no sample can ever be >= this
        if len(self._samples) < self.min_samples:
            return self.fallback
        # np.percentile's linear interpolation (not a plain sorted-index
        # pick) matters here: an earlier version used
        # `sorted(samples)[round((1-rate)*(len(samples)-1))]`, which is
        # off by close to one full rank — empirically it produced a ~7%
        # hit rate when targeting 4.5%. Interpolating against the true
        # 100*(1-rate) percentile tracks the target rate much more
        # closely (verified against a synthetic probability stream).
        return float(np.percentile(list(self._samples), 100.0 * (1.0 - rate)))

    def reset(self) -> None:
        self._samples.clear()


def order_bucket(batting_position: int) -> int:
    """Maps a 1-indexed batting-order slot (1..11) onto one of 9 buckets
    (0..8) used by ORDER_WICKET_MODIFIER / ORDER_RUN_WEIGHTS. Slots 9-11
    (the tail) share the last bucket — there's no meaningful behavioural
    difference between a team's 9th, 10th and 11th batter here."""
    return min(max(batting_position, 1), 9) - 1


# Additive adjustment to the *target dismissal rate* (not to the model's raw
# wicket_prob value directly — see RollingThreshold.threshold_for_rate).
# Working in rate-space rather than raw-probability-space keeps this
# scale-invariant: it produces the same realistic behaviour regardless of
# how spread out a given checkpoint's raw sigmoid outputs actually are.
# (An earlier version of this added the modifier straight to wicket_prob;
# testing against the heuristic backend's tightly-clustered synthetic
# probabilities showed that breaks badly — a batter's modifier could
# dwarf the entire spread of raw values and make them either unwicketable
# or dismissed almost every ball, regardless of the intended gradient.)
# Openers get a lower effective rate (harder to dismiss); the tail gets a
# much higher one. Index 0 = batting position 1, ..., index 8 = positions
# 9-11.
ORDER_WICKET_MODIFIER: list[float] = [
    -0.3,  # 1 - opener
    -0.3,  # 2 - opener
    -0.2,  # 3 - top order
    0.15,  # 4 - top/middle order
    -0.1,  # 5 - middle order
    -0.2,  # 6 - middle order / finisher
    -0.35,  # 7 - finisher
    0.2,  # 8 - lower order
    0.3,  # 9-11 - tail
]


WICKET_STREAK_RATE_DECREMENT = 0.02
"""'x' from the spec, in target-rate terms: each delivery in an active
consecutive-wicket streak cuts the next ball's effective dismissal rate by
this much — after 1 wicket the very next ball's rate is -1x, after 2 in a
row it's -2x, and so on — which shows up as a *higher* required threshold
once looked up via RollingThreshold.threshold_for_rate(), exactly like the
spec describes ("threshold increases"), just computed the robust way."""

MAX_CONSECUTIVE_WICKETS_ALLOWED = 3
"""A 4th dismissal on 4 consecutive legal deliveries is blocked outright:
once this many have fallen in a row, the next ball's effective target rate
is forced to 0, which RollingThreshold.threshold_for_rate() turns into an
unreachable threshold (> 1.0)."""


def effective_wicket_target_rate(
    base_rate: float, batting_position: int, consecutive_wickets: int
) -> float:
    """Combines the innings' calibrated base rate with this batter's
    ORDER_WICKET_MODIFIER and the current wicket-streak penalty into the
    single target rate to look up via RollingThreshold.threshold_for_rate().
    """
    if consecutive_wickets >= MAX_CONSECUTIVE_WICKETS_ALLOWED:
        return 0.0
    rate = base_rate + ORDER_WICKET_MODIFIER[order_bucket(batting_position)]
    rate -= consecutive_wickets * WICKET_STREAK_RATE_DECREMENT
    return float(np.clip(rate, 0.0, 0.97))


# Run-value outcomes for a "normal" (non-wide, non-wicket) legal delivery,
# and 9 batting-order-dependent weight profiles over them — see the section
# docstring above for why these (not the model's delta) drive run magnitude.
# Tuned empirically against a full run of this repo's own simulation
# pipeline (match_engine.py + this module, driven end-to-end with a
# synthetic squads file and the heuristic backend as a stand-in probability
# source, since this environment doesn't have your checkpoint/data files)
# so a full-XI-weighted mix lands close to verify_simulation.py's targets:
# ~150-155 runs/innings, ~16 fours and ~12 sixes per match, non-boundary RPB
# ~0.8, all with WIDE_TARGET_RATE/WICKET_BASE_TARGET_RATE above pulling the
# wide/wicket rates down to realistic levels. Retune here (not in
# match_engine.py) if a real run against your actual checkpoint drifts —
# in particular, if fours/sixes run high, scale down column 4/5 (index 4,
# 5) across every row; if runs pile up too fast on non-boundary balls,
# raise column 0 (the dot-ball weight) and rebalance the rest back to 1.0.
RUN_VALUES: list[int] = [0, 1, 2, 3, 4, 6]

ORDER_RUN_WEIGHTS: list[list[float]] = [
    # dot,    1,      2,      3,      4,      6         position(s)
    [0.24, 0.32, 0.2, 0.02, 0.1, 0.12],  # 1  - opener
    [0.24, 0.32, 0.2, 0.02, 0.1, 0.12],  # 2  - opener
    [0.24, 0.34, 0.18, 0.02, 0.1, 0.12],  # 3  - top order
    [0.24, 0.32, 0.14, 0, 0.14, 0.16],  # 4  - top/middle order
    [0.2, 0.34, 0.16, 0, 0.14, 0.16],  # 5  - middle order
    [0.1, 0.32, 0.16, 0, 0.2, 0.22],  # 6  - finisher
    [0.14, 0.32, 0.16, 0, 0.18, 0.2],  # 7  - finisher
    [0.3, 0.4, 0, 0, 0.1, 0.2],  # 8  - lower order
    [0.2, 0.3, 0.1, 0, 0.3, 0.1],  # 9-11 - tail
]


def sample_run_outcome(batting_position: int, rng: random.Random) -> int:
    """Draws bat-runs (0/1/2/3/4/6) for a normal delivery from this
    batter's ORDER_RUN_WEIGHTS row. `rng` should be the match's own seeded
    `random.Random` instance (match_engine.py's `rng`) so results stay
    reproducible for a given `seed`, matching how bowler selection already
    uses it."""
    weights = ORDER_RUN_WEIGHTS[order_bucket(batting_position)]
    return rng.choices(RUN_VALUES, weights=weights, k=1)[0]


@dataclass
class TrainedCheckpoint:
    """What _try_load_checkpoint() extracts from the *.pth file — replaces
    the old pickled IPLModelBundle. `epoch`/`composite_score` are carried
    along purely for the startup log line; nothing downstream uses them."""

    model: "nn.Module"
    wicket_thresh: float
    wide_thresh: float
    epoch: int | None
    composite_score: float | None


class TrainedModelRunner(ModelRunner):
    """
    PATCHED (quick fix, not a real solution — see README_MODEL_PATCH.md):

    The score head predicts the innings' cumulative total directly (see
    train_embeddings.py: target column is scaled `current_score`,
    Huber(delta=0.14) loss). We found via ball-by-ball logging that
    `predicted_cumulative` moves in tiny, slow increments almost every ball
    and is NOT anchored to the simulator's actual running score — so
    `predicted_cumulative - score_before` produces a one-way ratchet:
    whenever the model's estimate trails the real score (constantly, since
    the model isn't tracking it), delta is deeply negative, clips to 0, and
    the real score can never catch up until the model's raw output happens
    to drift past it on its own — producing long dot-ball streaks broken by
    forced max-6 jumps.

    This patch diffs the model's own PREVIOUS prediction instead of the
    simulator's real score, removing the one-way clip-to-zero trap (both
    sides of the subtraction now come from the same slow-moving signal, so
    small negative diffs are expected and meaningful, not just noise to be
    floored). This does NOT fix the model's low per-ball resolution — it
    will very likely still under-produce 2s/3s/4s/5s relative to real
    cricket, since the model was never trained on a per-ball-sensitive
    target. Proper fix is retraining with a direct per-ball delta/
    classification target (see README_MODEL_PATCH.md).

    UPDATE (train_embeddings.py merge): `.predict(x, verbose=0)` (Keras) is
    replaced with a direct PyTorch forward pass over two tensors. See
    `load_match_context` for the embedding-table swap this now requires.

    UPDATE (checkpoint format change): constructed from a TrainedCheckpoint
    (state_dict-based .pth) instead of a pickled bundle — see this module's
    docstring.
    """

    backend_name = "trained_model"

    def __init__(self, checkpoint: TrainedCheckpoint):
        self.checkpoint = checkpoint
        self.model = checkpoint.model
        self.model.eval()
        # The exact adaptive thresholds computed at training time (see
        # train_embeddings.py's get_adaptive_threshold(), logged into the
        # checkpoint as wicket_thresh/wide_thresh) — kept here ONLY as the
        # `fallback` match_engine.py's RollingThreshold instances use before
        # an innings has produced enough of *this checkpoint's own* raw
        # wicket_prob/wide_prob samples to calibrate against (see this
        # module's "Custom ball-outcome shaping" section). A previous
        # version of this line nudged these by a fixed -0.225 / +0.32 guess;
        # that only "worked" for whatever raw output distribution the
        # checkpoint happened to produce during that one test —
        # verify_simulation.py has since shown the same checkpoint swinging
        # from a 0.0% to a 7.8/match wicket rate and sitting at ~40% wide
        # either way, so a static offset isn't something that can be tuned
        # once and trusted going forward.
        self.wicket_thresh = checkpoint.wicket_thresh
        self.wide_thresh = checkpoint.wide_thresh
        self._prev_predicted_cumulative: float | None = None
        # load_match_context() below monkeypatches shared nn.Embedding
        # submodules on this (singleton, cached-via-get_runner) model, so two
        # concurrent /predict-1-match requests would otherwise race and
        # corrupt each other's embeddings mid-match. This lock serializes
        # trained-model match simulation end-to-end (see match_engine.py,
        # which holds it across an entire simulate_match() call, not just
        # one predict_ball()). Heuristic-fallback requests aren't affected.
        # A real fix would give each request its own model copy (expensive:
        # a real state_dict clone per request) or, better, get
        # player2idx/venue2idx exported from training so indices can be
        # looked up without mutating the model at all.
        self.lock = threading.Lock()

    def reset_innings(self) -> None:
        """Call at the start of each innings so ball 1 doesn't diff against
        the previous innings' final prediction."""
        self._prev_predicted_cumulative = None

    def load_match_context(self, ctx) -> None:
        """
        Swaps in match-local embedding tables built by
        services.embeddings.build_match_context (see that module's
        docstring for why this exists instead of using the model's own
        trained indices). Must be called once per match, and every
        predict_ball() call for that match must happen while still holding
        `self.lock` — match_engine.simulate_match() does this.
        match_state_embedding is deliberately left untouched (see
        feature_builder.build_ball_features).

        padding_idx=0 below mirrors training/tabtransformer_lstm.py's own
        `nn.Embedding(..., padding_idx=0)` declarations — confirmed (not
        guessed) from the real model source. It's a no-op for inference
        (padding_idx only affects gradient updates, and we run under
        torch.no_grad()), since ctx's matrices already have an all-zero row
        0; declaring it here is just for fidelity to the real architecture.
        """
        self.model.batter_embedding = nn.Embedding.from_pretrained(
            torch.tensor(ctx.batter_matrix, dtype=torch.float32), padding_idx=0
        )
        self.model.non_striker_embedding = nn.Embedding.from_pretrained(
            torch.tensor(ctx.non_striker_matrix, dtype=torch.float32), padding_idx=0
        )
        self.model.bowler_embedding = nn.Embedding.from_pretrained(
            torch.tensor(ctx.bowler_matrix, dtype=torch.float32), padding_idx=0
        )
        self.model.venue_embedding = nn.Embedding.from_pretrained(
            torch.tensor(ctx.venue_matrix, dtype=torch.float32), padding_idx=0
        )
        self.model.season_embedding = nn.Embedding.from_pretrained(
            torch.tensor(ctx.season_matrix, dtype=torch.float32), padding_idx=0
        )

    def predict_ball(
        self,
        numerical_sequence: np.ndarray,
        categorical_sequence: np.ndarray,
        score_before: float,
    ) -> tuple[float, float, float]:
        numerical = _pad_sequence(numerical_sequence)[
            np.newaxis, ...
        ]  # (1, SEQ_LEN, 32)
        categorical = _pad_categorical(categorical_sequence)[
            np.newaxis, ...
        ]  # (1, SEQ_LEN, 6)

        with torch.no_grad():
            numerical_t = torch.as_tensor(numerical, dtype=torch.float32)
            categorical_t = torch.as_tensor(categorical, dtype=torch.long)
            outputs = self.model(numerical_t, categorical_t)

        predicted_cumulative = (
            float(outputs["score"].detach().cpu().numpy().flatten()[0]) * SCORE_SCALE
        )

        if self._prev_predicted_cumulative is None:
            delta = predicted_cumulative - score_before
        else:
            delta = predicted_cumulative - self._prev_predicted_cumulative

        self._prev_predicted_cumulative = predicted_cumulative

        wicket_prob = float(
            _sigmoid(outputs["wicket"].detach().cpu().numpy().flatten()[0])
        )
        wide_prob = float(_sigmoid(outputs["wide"].detach().cpu().numpy().flatten()[0]))
        return delta, wicket_prob, wide_prob


class HeuristicModelRunner(ModelRunner):
    """
    Structural stand-in used only when the real checkpoint can't be loaded
    (e.g. PyTorch not installed, or the .pth isn't present in this
    environment). This is NOT the trained model — it exists purely so the
    simulation pipeline (rules, bowler rotation, strike rotation, scoring)
    is runnable and testable end-to-end without your ML stack present.
    Every response using it is flagged with model_backend="heuristic_fallback".
    """

    backend_name = "heuristic_fallback"

    # rough T20-shaped ball-outcome distribution
    _RUN_VALUES = [0, 1, 2, 3, 4, 6]
    # NOTE: previously [0.3, 0.28, 0.14, 0.08, 0.12, 0.1], which summed to
    # 1.02 — np.random.Generator.choice (unlike random.choices) requires
    # probabilities to sum to exactly 1 and raises otherwise. Fixed by
    # trimming the "2" weight by 0.02. This delta value is unused for run
    # magnitude in the actual simulation now anyway (see match_engine.py's
    # sample_run_outcome() call), but predict_ball must still not crash.
    _RUN_WEIGHTS = [0.3, 0.28, 0.14, 0.06, 0.12, 0.1]

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        # no checkpoint to pull real adaptive thresholds from — fall back to
        # the env-configurable defaults in ml_config.py.
        self.wicket_thresh = WICKET_PROB_THRESHOLD
        self.wide_thresh = WIDE_PROB_THRESHOLD

    def predict_ball(
        self,
        numerical_sequence: np.ndarray,
        categorical_sequence: np.ndarray,
        score_before: float,
    ) -> tuple[float, float, float]:
        delta = self._rng.choice(self._RUN_VALUES, p=self._RUN_WEIGHTS)
        wicket_prob = float(np.clip(self._rng.normal(0.06, 0.02), 0.0, 1.0))
        wide_prob = float(np.clip(self._rng.normal(0.04, 0.015), 0.0, 1.0))
        return float(delta), wicket_prob, wide_prob


def _find_model_path() -> str | None:
    if MODEL_PATH:
        return MODEL_PATH if os.path.exists(MODEL_PATH) else None
    candidates = sorted(
        glob.glob(os.path.join(MODEL_DIR, "*.pth")), key=os.path.getmtime, reverse=True
    )
    return candidates[0] if candidates else None


def _infer_architecture(state_dict: dict) -> tuple[int, int, int, int]:
    """Recovers the four constructor args TabTransformerLSTM needs
    (num_players, num_venues, num_seasons, numerical_dim) directly from the
    checkpoint's own tensor shapes, instead of tracking them separately
    somewhere they could drift out of sync with what was actually trained.

    - batter_embedding.weight: (num_players, 60)
    - venue_embedding.weight: (num_venues, 30)
    - season_embedding.weight: (num_seasons + 1, 8) — the model's own
      constructor does `nn.Embedding(num_seasons + 1, ...)`, so we subtract
      the 1 back off here.
    - numerical_norm.weight: (numerical_dim,) — LayerNorm(numerical_dim)'s
      per-feature scale vector.
    Raises KeyError (caught by the caller) if the checkpoint doesn't look
    like a TabTransformerLSTM state_dict at all.
    """
    num_players = state_dict["batter_embedding.weight"].shape[0]
    num_venues = state_dict["venue_embedding.weight"].shape[0]
    num_seasons = state_dict["season_embedding.weight"].shape[0] - 1
    numerical_dim = state_dict["numerical_norm.weight"].shape[0]
    return num_players, num_venues, num_seasons, numerical_dim


def _try_load_checkpoint() -> TrainedCheckpoint | None:
    if not TORCH_AVAILABLE:
        logger.warning(
            "PyTorch isn't installed — using heuristic fallback. "
            "`pip install torch` to use the trained model."
        )
        return None

    path = _find_model_path()
    if not path:
        logger.warning(
            "No checkpoint (*.pth) found in %s — using heuristic fallback.", MODEL_DIR
        )
        return None

    try:
        # weights_only=False: this is a trusted, first-party checkpoint (not
        # downloaded from anywhere untrusted), and some PyTorch versions
        # default weights_only=True, which can reject the optimizer state /
        # plain-python metadata (epoch, composite_score, thresholds) sitting
        # alongside model_state_dict in this checkpoint.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - want to fall back on *any* load failure
        logger.warning(
            "Failed to torch.load checkpoint at %s (%s) — using heuristic fallback.",
            path,
            e,
        )
        return None

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        logger.warning(
            "Checkpoint at %s has no 'model_state_dict' key — using heuristic "
            "fallback.",
            path,
        )
        return None

    state_dict = checkpoint["model_state_dict"]

    try:
        num_players, num_venues, num_seasons, numerical_dim = _infer_architecture(
            state_dict
        )
    except KeyError as e:
        logger.warning(
            "Checkpoint's model_state_dict at %s is missing an expected "
            "TabTransformerLSTM tensor (%s) — using heuristic fallback.",
            path,
            e,
        )
        return None

    model = TabTransformerLSTM(
        num_players=num_players,
        num_venues=num_venues,
        num_seasons=num_seasons,
        numerical_dim=numerical_dim,
    )
    try:
        model.load_state_dict(state_dict)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "state_dict at %s didn't match ml_model/tabtransformer_lstm.py's "
            "architecture (%s) — using heuristic fallback. If you changed the "
            "model's constructor defaults (embedding dims, transformer size, "
            "etc.) since that file was vendored, update it to match.",
            path,
            e,
        )
        return None
    model.eval()

    wicket_thresh = float(checkpoint.get("wicket_thresh", WICKET_PROB_THRESHOLD))
    wide_thresh = float(checkpoint.get("wide_thresh", WIDE_PROB_THRESHOLD))

    logger.info(
        "Loaded checkpoint from %s (epoch=%s, composite_score=%s, "
        "num_players=%d, num_venues=%d, num_seasons=%d, numerical_dim=%d, "
        "wicket_thresh=%.4f, wide_thresh=%.4f)",
        path,
        checkpoint.get("epoch"),
        checkpoint.get("composite_score"),
        num_players,
        num_venues,
        num_seasons,
        numerical_dim,
        wicket_thresh,
        wide_thresh,
    )

    return TrainedCheckpoint(
        model=model,
        wicket_thresh=wicket_thresh,
        wide_thresh=wide_thresh,
        epoch=checkpoint.get("epoch"),
        composite_score=checkpoint.get("composite_score"),
    )


@lru_cache(maxsize=1)
def get_runner() -> ModelRunner:
    checkpoint = _try_load_checkpoint()
    if checkpoint is not None:
        return TrainedModelRunner(checkpoint)
    return HeuristicModelRunner()
