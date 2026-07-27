"""
The ball-by-ball simulator. Orchestrates squads, embeddings, recent-form,
the model runner, and bowler selection under standard T20 rules:

  - Max MAX_OVERS_PER_BOWLER overs per bowler.
  - No bowler bowls two consecutive overs.
  - Strike rotates on odd runs off a legal ball.
  - Strike rotates again at the end of every over (ends change) — combined
    with the odd-run rule above exactly as it works in real cricket.
  - Wide = +1 run, does not consume a legal ball, no strike rotation.
  - Wicket = batter out, next batter in from the batting order; the model
    only signals *that* a wicket fell (not how), so every dismissal is
    recorded generically as "b <bowler>" — a documented simplification.
  - Chase ends immediately once the target is reached, even mid-over.

Only two extras types exist because the model only has wicket/wide heads
(no no-ball/bye/leg-bye output) — matching train.py's target columns
exactly (`current_score`, `is_wicket_target`, `isWide_target`).
"""
from __future__ import annotations

import contextlib
import logging
import random

import numpy as np

from ml_config import (
    MAX_OVERS_PER_BOWLER,
    RANDOM_SEED,
    TOTAL_OVERS,
    SEASON,
)
from services import embeddings
from services import historical_form
from services.bowler_selector import BowlerUsage, pick_next_bowler
from services.feature_builder import BallContext, build_ball_features, _rr_momentum
from services.match_state import BatterInnings, BowlerInnings, InningsState, MatchResult
from services.model_runner import get_runner
from services.squads import Player, Team, get_team

logger = logging.getLogger(__name__)

def _resolve_toss(team_a: str, team_b: str, rng: random.Random) -> tuple[str, str]:
    winner = rng.choice([team_a, team_b])
    decision = rng.choice(["bat", "bowl"])
    return winner, decision


def _new_innings(
    inning_no: int,
    batting_team: Team,
    bowling_team: Team,
    target: int | None,
) -> InningsState:
    striker = BatterInnings(player=batting_team.batting_order[0])
    non_striker = BatterInnings(player=batting_team.batting_order[1])
    innings = InningsState(
        inning_no=inning_no,
        batting_team=batting_team.code,
        bowling_team=bowling_team.code,
        batting_order=batting_team.batting_order,
        bowling_candidates=bowling_team.bowling_pool,
        target=target,
        striker=striker,
        non_striker=non_striker,
    )
    innings.batters[striker.player.name] = striker
    innings.batters[non_striker.player.name] = non_striker
    return innings


def _get_or_create_bowler(innings: InningsState, player: Player) -> BowlerInnings:
    if player.name not in innings.bowlers:
        innings.bowlers[player.name] = BowlerInnings(player=player)
    return innings.bowlers[player.name]


def _bring_in_next_batter(innings: InningsState) -> BatterInnings | None:
    if innings.next_batter_idx >= len(innings.batting_order):
        return None
    player = innings.batting_order[innings.next_batter_idx]
    innings.next_batter_idx += 1
    batter = BatterInnings(player=player)
    innings.batters[player.name] = batter
    return batter


def _swap_strike(innings: InningsState) -> None:
    innings.striker, innings.non_striker = innings.non_striker, innings.striker


def _simulate_innings(
    innings: InningsState,
    venue: str,
    toss_winner: str,
    resolver: embeddings.EmbeddingResolver,
    match_ctx: embeddings.MatchEmbeddingContext,
    runner,
    rng: random.Random,
) -> None:
    if hasattr(runner, "reset_innings"):
        runner.reset_innings()

    # Runaway-model safety valve, discovered while testing the checkpoint
    # loading change: wides don't consume a legal ball
    # (innings.balls_in_current_over never increments), so if
    # wide_prob >= runner.wide_thresh on nearly every ball — e.g. an
    # untrained/miscalibrated checkpoint, or wicket_thresh/wide_thresh that
    # don't actually match this model's real output distribution — the
    # inner ball loop below can spin forever and hang the request. A real,
    # properly-thresholded model rarely needs more than ~135-150 total
    # attempts for a 120-legal-ball innings (wides are typically ~5% of
    # deliveries), so this cap is generous and shouldn't trip in normal use.
    MAX_BALL_ATTEMPTS = 150
    ball_attempts = 0

    while not innings.is_complete():
        # --- pick this over's bowler ---
        bowler_player = pick_next_bowler(
            innings.bowling_candidates,
            innings.bowler_usage,
            innings.current_over,
            rng=rng,
        )
        current_bowler = _get_or_create_bowler(innings, bowler_player)
        innings.balls_in_current_over = 0

        while innings.balls_in_current_over < 6:
            if innings.is_complete():
                break
            if innings.striker is None:
                break  # all out mid-over, handled by outer loop exit

            ball_attempts += 1
            if ball_attempts > MAX_BALL_ATTEMPTS:
                logger.warning(
                    "Innings %d hit MAX_BALL_ATTEMPTS (%d) without completing "
                    "120 legal balls — wide_prob is very likely stuck above "
                    "runner.wide_thresh=%.4f on nearly every ball (miscalibrated "
                    "or untrained checkpoint?). Ending the innings early instead "
                    "of hanging the request.",
                    innings.inning_no,
                    MAX_BALL_ATTEMPTS,
                    runner.wide_thresh,
                )
                return

            batter_form = historical_form.get_batter_form(innings.striker.player.name)
            bowler_form = historical_form.get_bowler_form(bowler_player.name)

            current_run_rate = (
                innings.score / (innings.legal_balls / 6.0) if innings.legal_balls else 0.0
            )
            pct_target = (
                (innings.score / innings.target * 100.0)
                if innings.target
                else 0.0
            )
            required_rr = 0.0
            if innings.target is not None and innings.legal_balls < 120:
                overs_left = innings.balls_remaining() / 6.0
                if overs_left > 0:
                    required_rr = (innings.target - innings.score) / overs_left

            ctx = BallContext(
                inning_no=innings.inning_no,
                over_idx=innings.current_over,
                ball_in_over=innings.balls_in_current_over + 1,
                legal_balls_before=innings.legal_balls,
                wickets_before=innings.wickets,
                target=innings.target,
                striker_name=innings.striker.player.name,
                non_striker_name=innings.non_striker.player.name,
                bowler_name=bowler_player.name,
                bowler_is_pacer=bool(bowler_player.is_pacer),
                venue=venue,
                toss_winner_is_batting_team=(toss_winner == innings.batting_team),
                batter_history_matches=batter_form.history_matches,
                batter_last_runs=batter_form.last_runs,
                batter_last_balls=batter_form.last_balls,
                bowler_history_matches=bowler_form.history_matches,
                bowler_last_runs_conceded=bowler_form.last_runs_conceded,
                bowler_last_balls_bowled=bowler_form.last_balls_bowled,
                rr_momentum=_rr_momentum(innings),
                season=SEASON,
            )
            numeric_row, categorical_row = build_ball_features(
                ctx, current_run_rate, pct_target, required_rr, match_ctx
            )

            # sequence buffers for this innings, most-recent-last
            innings.numerical_buffer.append(numeric_row)
            innings.categorical_buffer.append(categorical_row)
            numerical_sequence = np.stack(innings.numerical_buffer)
            categorical_sequence = np.stack(innings.categorical_buffer)

            delta, wicket_prob, wide_prob = runner.predict_ball(
                numerical_sequence, categorical_sequence, score_before=float(innings.score)
            )

            if wide_prob >= runner.wide_thresh:
                innings.score += 1
                current_bowler.runs_conceded += 1
                # wide: no legal ball consumed, no strike change, bowler continues
                continue
            
            if wicket_prob >= runner.wicket_thresh:
                innings.striker.balls += 1
                innings.striker.is_out = True
                innings.striker.dismissal = f"b {bowler_player.name}"
                innings.wickets += 1
                current_bowler.wickets += 1
                innings.legal_balls += 1
                innings.balls_in_current_over += 1
                innings.recent_ball_runs.append(0)

                next_batter = _bring_in_next_batter(innings)
                innings.striker = next_batter  # may be None if all out
                if innings.striker is None:
                    break
                continue

            a = round(delta)
            options = [0, 1, 2, 3]
            weights = [5, 2, 1, 0.01]
            s = random.choices(options, weights=weights, k=1)[0]
            runs = int(np.clip((a+s), 0, 6))
            innings.score += runs
            innings.striker.runs += runs
            innings.striker.balls += 1
            if runs == 4:
                innings.striker.fours += 1
            elif runs == 6:
                innings.striker.sixes += 1
            current_bowler.runs_conceded += runs
            innings.legal_balls += 1
            innings.balls_in_current_over += 1
            innings.recent_ball_runs.append(runs)

            if runs % 2 == 1:
                _swap_strike(innings)

        # --- over complete (or innings ended mid-over) ---
        current_bowler.legal_balls += innings.balls_in_current_over
        innings.bowler_usage[bowler_player.name] = BowlerUsage(
            overs_bowled=current_bowler.overs_bowled,
            last_over_bowled=innings.current_over,
        )
        if innings.balls_in_current_over == 6 and not innings.is_complete():
            _swap_strike(innings)
        innings.current_over += 1

        if innings.current_over >= TOTAL_OVERS:
            break


def _serialize_innings(innings: InningsState) -> dict:
    batters = [b.to_dict() for b in innings.batters.values()]
    bowlers = [b.to_dict() for b in innings.bowlers.values()]
    overs_str = f"{innings.legal_balls // 6}.{innings.legal_balls % 6}"
    return {
        "inning": innings.inning_no,
        "batting_team": innings.batting_team,
        "bowling_team": innings.bowling_team,
        "target": innings.target,
        "total": {
            "runs": innings.score,
            "wickets": innings.wickets,
            "overs": overs_str,
        },
        "batting": batters,
        "bowling": bowlers,
    }


def simulate_match(
    team_a_code: str,
    team_b_code: str,
    venue: str | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    seed: int | None = None,
) -> MatchResult:
    team_a = get_team(team_a_code)
    team_b = get_team(team_b_code)

    effective_seed = seed if seed is not None else (
        int(RANDOM_SEED) if RANDOM_SEED else None
    )
    rng = random.Random(effective_seed)
    emb_rng = np.random.default_rng(effective_seed)
    resolver = embeddings.EmbeddingResolver(emb_rng)
    runner = get_runner()

    if venue is None:
        venues = embeddings.list_venues()
        venue = rng.choice(venues) if venues else "Neutral Venue"

    if toss_winner is None or toss_decision is None:
        toss_winner, toss_decision = _resolve_toss(team_a.code, team_b.code, rng)

    if toss_decision == "bat":
        first_batting, first_bowling = (
            (team_a, team_b) if toss_winner == team_a.code else (team_b, team_a)
        )
    else:
        first_bowling, first_batting = (
            (team_a, team_b) if toss_winner == team_a.code else (team_b, team_a)
        )

    # Match-local embedding tables (see services/embeddings.py's module
    # docstring "UPDATE" section for why these are needed instead of the
    # model's own trained indices) — built once, shared by both innings
    # since the same 22+ players and venue apply throughout the match.
    batting_pool_names = [p.name for team in (team_a, team_b) for p in team.batting_order]
    bowling_pool_names = [p.name for team in (team_a, team_b) for p in team.bowling_pool]
    match_ctx = embeddings.build_match_context(
        batting_pool_names, bowling_pool_names, venue, resolver
    )

    # TrainedModelRunner.load_match_context swaps shared embedding
    # submodules on the (singleton) model — see its docstring. This lock
    # (a no-op contextlib.nullcontext for HeuristicModelRunner, which has
    # no `.lock`) serializes an entire match's simulation so a concurrent
    # request can't repatch those submodules mid-match.
    match_lock = getattr(runner, "lock", None)
    with match_lock if match_lock is not None else contextlib.nullcontext():
        if hasattr(runner, "load_match_context"):
            runner.load_match_context(match_ctx)

        innings1 = _new_innings(1, first_batting, first_bowling, target=None)
        _simulate_innings(innings1, venue, toss_winner, resolver, match_ctx, runner, rng)

        target = innings1.score + 1
        innings2 = _new_innings(2, first_bowling, first_batting, target=target)
        _simulate_innings(innings2, venue, toss_winner, resolver, match_ctx, runner, rng)

    if innings2.score >= target:
        winner = innings2.batting_team
        margin = 10 - innings2.wickets
        result = f"{winner} won by {margin} wicket{'s' if margin != 1 else ''}"
    elif innings2.score == innings1.score:
        winner = None
        result = "Match tied"
    else:
        winner = innings1.batting_team
        margin = innings1.score - innings2.score
        result = f"{winner} won by {margin} run{'s' if margin != 1 else ''}"

    return MatchResult(
        team_a=team_a.code,
        team_b=team_b.code,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        innings=[_serialize_innings(innings1), _serialize_innings(innings2)],
        result=result,
        winner=winner,
        model_backend=runner.backend_name,
    )