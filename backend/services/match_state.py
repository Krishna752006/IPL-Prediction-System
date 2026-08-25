from __future__ import annotations

from dataclasses import dataclass, field

from services.bowler_selector import BowlerUsage
from services.squads import Player


@dataclass
class BatterInnings:
    player: Player
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False
    dismissal: str | None = None  # e.g. "b Matt Henry"

    def to_dict(self) -> dict:
        return {
            "name": self.player.name,
            "runs": self.runs,
            "balls": self.balls,
            "fours": self.fours,
            "sixes": self.sixes,
            "strike_rate": (
                round(self.runs / self.balls * 100, 2) if self.balls else 0.0
            ),
            "out": self.is_out,
            "dismissal": self.dismissal,
        }


@dataclass
class BowlerInnings:
    player: Player
    legal_balls: int = 0
    runs_conceded: int = 0
    wickets: int = 0

    @property
    def overs_bowled(self) -> float:
        return self.legal_balls / 6.0

    def to_dict(self) -> dict:
        overs_str = f"{self.legal_balls // 6}.{self.legal_balls % 6}"
        econ = (self.runs_conceded / self.legal_balls * 6) if self.legal_balls else 0.0
        return {
            "name": self.player.name,
            "overs": overs_str,
            "runs_conceded": self.runs_conceded,
            "wickets": self.wickets,
            "economy": round(econ, 2),
        }


@dataclass
class InningsState:
    inning_no: int  # 1 or 2
    batting_team: str
    bowling_team: str
    batting_order: list[Player]
    bowling_candidates: list[Player]  # eligible bowlers for bowling_team
    target: int | None = None  # None for inning 1

    score: int = 0
    wickets: int = 0
    legal_balls: int = 0  # 0..120

    next_batter_idx: int = 2  # index into batting_order for the next-in
    striker: BatterInnings | None = None
    non_striker: BatterInnings | None = None
    batters: dict[str, BatterInnings] = field(default_factory=dict)
    bowlers: dict[str, BowlerInnings] = field(default_factory=dict)
    bowler_usage: dict[str, BowlerUsage] = field(default_factory=dict)

    current_bowler: BowlerInnings | None = None
    current_over: int = 0  # 0-indexed
    balls_in_current_over: int = 0
    recent_ball_runs: list[int] = field(
        default_factory=list
    )  # legal-ball runs, for rr_momentum
    # Per-ball feature tensors, most-recent-last (split in two since the
    # TabTransformerLSTM trained by train_embeddings.py takes numerical
    # features and categorical embedding indices as separate inputs — see
    # services/feature_builder.py).
    numerical_buffer: list = field(default_factory=list)
    categorical_buffer: list = field(default_factory=list)

    def is_complete(self) -> bool:
        if self.wickets >= 10:
            return True
        if self.legal_balls >= 120:
            return True
        if self.target is not None and self.score >= self.target:
            return True
        return False

    def overs_completed_float(self) -> float:
        return self.legal_balls / 6.0

    def balls_remaining(self) -> int:
        return max(0, 120 - self.legal_balls)


@dataclass
class MatchResult:
    team_a: str
    team_b: str
    venue: str
    toss_winner: str
    toss_decision: str
    innings: list[dict]  # serialized scorecards, in batting order
    result: str
    winner: str | None
    model_backend: str  # "trained_model" or "heuristic_fallback"

    def to_dict(self) -> dict:
        return {
            "tournament_context": {
                "team_a": self.team_a,
                "team_b": self.team_b,
                "venue": self.venue,
                "toss_winner": self.toss_winner,
                "toss_decision": self.toss_decision,
            },
            "innings": self.innings,
            "result": self.result,
            "winner": self.winner,
            "model_backend": self.model_backend,
        }
