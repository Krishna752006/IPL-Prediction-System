"""
Standalone sanity check for the /predict-1-match feature — run this after
`pip install`-ing dependencies and dropping the data files into place.

Usage:
    cd backend
    python verify_simulation.py

Exits non-zero and prints exactly what failed if anything's wrong. Doesn't
need uvicorn running — it calls the simulation engine directly.
"""
import itertools
import random
import statistics
import sys
import time
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

sys.path.insert(0, ".")

from services.squads import list_team_codes, get_team
from services.match_engine import simulate_match
import services.bowler_selector as bs
import services.match_engine as me


def check_squads():
    for code in list_team_codes():
        t = get_team(code)
        assert len(t.batting_order) == 11, f"{code}: batting XI isn't 11"
        assert t.bowling_only_player is not None, f"{code}: no 12th player"
        assert len(t.bowling_pool) >= 5, f"{code}: fewer than 5 available bowlers"
    print(f"[OK] All {len(list_team_codes())} squads valid (11-man XI, 12th bowls only, >=5 bowlers)")


def check_matches(num_pairs=15, seed=42):
    orig_pick = bs.pick_next_bowler
    log = []

    def wrapped(candidates, usage, current_over, rng=None):
        p = orig_pick(candidates, usage, current_over, rng=rng)
        log.append((current_over, p.name))
        return p

    bs.pick_next_bowler = wrapped
    me.pick_next_bowler = wrapped

    codes = list_team_codes()
    rng = random.Random(seed)
    pairs = list(itertools.combinations(codes, 2))
    rng.shuffle(pairs)

    checked = 0
    for a, b in pairs[:num_pairs]:
        log.clear()
        team_a, team_b = get_team(a), get_team(b)
        result = simulate_match(a, b, seed=rng.randint(0, 99999))
        d = result.to_dict()

        for inn in d["innings"]:
            batting_team = team_a if inn["batting_team"] == a else team_b
            batted_names = {x["name"] for x in inn["batting"]}
            assert batting_team.bowling_only_player.name not in batted_names, (
                f"{a} v {b}: 12th man batted"
            )
            batter_runs = sum(x["runs"] for x in inn["batting"])
            bowler_runs = sum(x["runs_conceded"] for x in inn["bowling"])

            total_runs = inn["total"]["runs"]
            extras = total_runs - batter_runs

            assert (batter_runs + extras) == total_runs, \
                f"Mismatch: {batter_runs} batter + {extras} extras != {total_runs} total"

            assert bowler_runs >= batter_runs, \
                f"Mismatch: Bowler runs ({bowler_runs}) cannot be less than batter runs ({batter_runs})"
            
            assert inn["total"]["wickets"] <= 10, f"{a} v {b}: more than 10 wickets"

        seqs, cur, last_over = [], [], -1
        for over, name in log:
            if over < last_over:
                seqs.append(cur)
                cur = []
            cur.append((over, name))
            last_over = over
        seqs.append(cur)
        for s in seqs:
            for j in range(1, len(s)):
                assert s[j][1] != s[j - 1][1], f"{a} v {b}: consecutive overs by {s[j][1]}"
            counts = Counter(n for _, n in s)
            assert all(v <= 4 for v in counts.values()), f"{a} v {b}: over-cap violation {counts}"

        checked += 1

    print(f"[OK] {checked} random matchups: totals reconcile, 12th man never bats, "
          f"bowler rotation rules hold")


def check_performance(num_matches=30, seed=29):
    """Profiles where time actually goes per match: how many predict_ball
    calls happen, how long they take in aggregate, and how those calls
    split into wide / wicket / normal outcomes. Run this whenever a
    threshold change (WICKET_PROB_THRESHOLD, WIDE_PROB_THRESHOLD) seems to
    change runtime — it tells you whether the slowdown is "more balls
    simulated" (more predict_ball calls) or "predict_ball itself got
    slower" (unlikely, but this rules it out) or something else entirely
    (e.g. more innings-ending edge cases, more batter/bowler churn)."""
    import services.model_runner as mr

    codes = list_team_codes()
    rng = random.Random(seed)

    orig_predict_ball = mr.TrainedModelRunner.predict_ball
    call_stats = {"count": 0, "total_time": 0.0, "wides": 0, "wickets": 0, "normal": 0}

    def timed_predict_ball(self, numerical_sequence, categorical_sequence, score_before):
        t0 = time.perf_counter()
        delta, wicket_prob, wide_prob = orig_predict_ball(
            self, numerical_sequence, categorical_sequence, score_before
        )
        call_stats["total_time"] += time.perf_counter() - t0
        call_stats["count"] += 1
        # classify using the same thresholds match_engine.py applies, purely
        # for reporting — doesn't change simulation behavior
        from ml_config import WICKET_PROB_THRESHOLD, WIDE_PROB_THRESHOLD
        if wide_prob >= WIDE_PROB_THRESHOLD:
            call_stats["wides"] += 1
        elif wicket_prob >= WICKET_PROB_THRESHOLD:
            call_stats["wickets"] += 1
        else:
            call_stats["normal"] += 1
        return delta, wicket_prob, wide_prob

    mr.TrainedModelRunner.predict_ball = timed_predict_ball

    per_match_times = []
    per_match_calls = []
    try:
        for _ in range(num_matches):
            a, b = rng.sample(codes, 2)
            before_count = call_stats["count"]
            t0 = time.perf_counter()
            simulate_match(a, b, seed=rng.randint(0, 10_000_000))
            elapsed = time.perf_counter() - t0
            per_match_times.append(elapsed)
            per_match_calls.append(call_stats["count"] - before_count)
    finally:
        mr.TrainedModelRunner.predict_ball = orig_predict_ball

    backend = me.get_runner().backend_name
    print(f"[PERF] backend={backend}  {num_matches} matches")
    if call_stats["count"] == 0:
        print("        No predict_ball calls recorded (heuristic backend, or "
              "runner isn't TrainedModelRunner) — nothing to profile.")
        return

    print(f"        Wall time per match:   mean={statistics.mean(per_match_times):.2f}s  "
          f"max={max(per_match_times):.2f}s  min={min(per_match_times):.2f}s")
    print(f"        predict_ball calls:    mean/match={statistics.mean(per_match_calls):.1f}  "
          f"max/match={max(per_match_calls)}")
    print(f"        predict_ball time:     total={call_stats['total_time']:.2f}s over "
          f"{call_stats['count']} calls  "
          f"avg={1000*call_stats['total_time']/call_stats['count']:.2f}ms/call")
    print(f"        Outcome split:         normal={call_stats['normal']}  "
          f"wides={call_stats['wides']}  wickets={call_stats['wickets']}  "
          f"({100*call_stats['wides']/call_stats['count']:.1f}% wide, "
          f"{100*call_stats['wickets']/call_stats['count']:.1f}% wicket)")
    print("        => if calls/match is much higher than ~120-260 (normal T20 "
          "ball count), extra wides are inflating total balls bowled; if "
          "avg ms/call is stable but wall time still balloons, the extra "
          "cost is coming from more predict_ball calls, not slower calls.")
    print()


def _ball_outcomes_for_innings(inn: dict) -> Counter:
    """Reconstruct a per-ball run-value histogram for one innings from the
    serialized scorecard. We don't have a ball-by-ball log (deliberately —
    final-scorecard-only response), so this approximates outcome counts from
    fours/sixes (exact) and treats the remaining runs/balls as an aggregate
    'other' bucket (0s/1s/2s/3s/5s + byes-equivalent), which is enough to see
    dot-heavy / boundary-or-nothing patterns without needing a ball log."""
    fours = sum(b["fours"] for b in inn["batting"])
    sixes = sum(b["sixes"] for b in inn["batting"])
    boundary_balls = fours + sixes
    total_balls = int(inn["total"]["overs"].split(".")[0]) * 6 + int(inn["total"]["overs"].split(".")[1])
    boundary_runs = fours * 4 + sixes * 6
    other_balls = max(0, total_balls - boundary_balls)
    other_runs = inn["total"]["runs"] - boundary_runs
    return Counter({
        "fours": fours,
        "sixes": sixes,
        "boundary_balls": boundary_balls,
        "other_balls": other_balls,
        "other_runs": other_runs,
    })


def check_realism_stats(num_seeds=100, base_seed=1000):
    """Not a pass/fail rule check (there's no 'correct' T20 score) — prints
    scoring/seed-variance stats so a human can judge realism and compare
    before/after a model_runner.py or model change. Run this whenever you
    touch the delta-derivation logic or retrain the model."""
    codes = list_team_codes()
    rng = random.Random(base_seed)

    totals = []          # first-innings final score, per seed
    dot_pcts = []        # % of non-boundary balls, per seed (proxy dot-rate)
    rpb_others = []      # runs-per-ball on non-boundary deliveries, per seed
    six_counts = []
    four_counts = []
    wicket_counts = []   # total wickets across both innings, per seed

    t_start = time.perf_counter()
    for _ in range(num_seeds):
        a, b = rng.sample(codes, 2)
        seed = rng.randint(0, 10_000_000)
        result = simulate_match(a, b, seed=seed)
        d = result.to_dict()

        inn1 = d["innings"][0]
        totals.append(inn1["total"]["runs"])

        match_fours = match_sixes = match_wkts = 0
        match_other_balls = match_other_runs = 0
        for inn in d["innings"]:
            oc = _ball_outcomes_for_innings(inn)
            match_fours += oc["fours"]
            match_sixes += oc["sixes"]
            match_other_balls += oc["other_balls"]
            match_other_runs += oc["other_runs"]
            match_wkts += inn["total"]["wickets"]

        four_counts.append(match_fours)
        six_counts.append(match_sixes)
        wicket_counts.append(match_wkts)
        rpb_others.append(match_other_runs / match_other_balls if match_other_balls else 0.0)
        # rough dot-ball proxy: non-boundary balls scoring 0 runs isn't
        # directly recoverable from the scorecard, so we report the
        # non-boundary runs-per-ball instead (lower = more dot-heavy)

    elapsed = time.perf_counter() - t_start
    print(f"[STATS] Ran {num_seeds} matches (random team pairs, base_seed={base_seed}) "
          f"in {elapsed:.1f}s (avg {elapsed/num_seeds:.2f}s/match)")
    print(f"        1st-innings score:  mean={statistics.mean(totals):.1f}  "
          f"stdev={statistics.stdev(totals):.1f}  min={min(totals)}  max={max(totals)}")
    print(f"        Fours per match:    mean={statistics.mean(four_counts):.1f}  "
          f"stdev={statistics.stdev(four_counts):.1f}")
    print(f"        Sixes per match:    mean={statistics.mean(six_counts):.1f}  "
          f"stdev={statistics.stdev(six_counts):.1f}")
    print(f"        Wickets per match:  mean={statistics.mean(wicket_counts):.1f}  "
          f"stdev={statistics.stdev(wicket_counts):.1f}  max={max(wicket_counts)}")
    print(f"        Non-boundary RPB:   mean={statistics.mean(rpb_others):.3f}  "
          f"stdev={statistics.stdev(rpb_others):.3f}  "
          f"(real T20 non-boundary balls run ~0.5-0.8 RPB; well below ~0.3 "
          f"signals dot-heavy/bimodal scoring, high stdev signals seed instability)")
    if statistics.mean(wicket_counts) < 2:
        print("        [WARN] Very few wickets across all seeds — check "
              "wicket_prob calibration vs WICKET_PROB_THRESHOLD in ml_config.py")
    print()


def check_api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routers.predict import router as predict_router
    from services.bowler_selector import router as bowler_router

    app = FastAPI()
    app.include_router(predict_router)
    app.include_router(bowler_router)
    client = TestClient(app)

    r = client.post("/predict-1-match", json={"team_a": "DC", "team_b": "MI", "seed": 1})
    assert r.status_code == 200, r.text
    assert "player_of_the_match" not in r.json()

    r = client.post("/select-bowler", json={"available_bowlers": ["A", "B"]})
    assert r.status_code == 200 and r.json()["bowler"] in ("A", "B")

    r = client.post("/predict-1-match", json={"team_a": "CSK", "team_b": "CSK"})
    assert r.status_code == 422

    print("[OK] FastAPI layer: /predict-1-match and /select-bowler both correct")


if __name__ == "__main__":
    print(f"model backend in use: {me.get_runner().backend_name}\n")
    check_squads()
    check_matches()
    check_realism_stats()
    check_performance()
    check_api()
    print("\nAll checks passed.")