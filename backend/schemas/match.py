from __future__ import annotations

from pydantic import BaseModel, Field
from services.squads import list_team_codes


class PredictMatchRequest(BaseModel):
    team_a: str = Field(..., description=f"Squad code, one of {list_team_codes()}")
    team_b: str = Field(..., description=f"Squad code, one of {list_team_codes()}")
    venue: str | None = Field(
        None, description="Optional venue name; picked randomly if omitted"
    )
    toss_winner: str | None = Field(
        None, description="Optional — must equal team_a or team_b. Random if omitted."
    )
    toss_decision: str | None = Field(
        None, description="'bat' or 'bowl'. Random if omitted."
    )
    seed: int | None = Field(
        None, description="Optional RNG seed for a reproducible simulation"
    )


class BatterLine(BaseModel):
    name: str
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    out: bool
    dismissal: str | None


class BowlerLine(BaseModel):
    name: str
    overs: str
    runs_conceded: int
    wickets: int
    economy: float


class InningsTotal(BaseModel):
    runs: int
    wickets: int
    overs: str


class InningsScorecard(BaseModel):
    inning: int
    batting_team: str
    bowling_team: str
    target: int | None
    total: InningsTotal
    batting: list[BatterLine]
    bowling: list[BowlerLine]


class TournamentContext(BaseModel):
    team_a: str
    team_b: str
    venue: str
    toss_winner: str
    toss_decision: str


class PredictMatchResponse(BaseModel):
    tournament_context: TournamentContext
    innings: list[InningsScorecard]
    result: str
    winner: str | None
    model_backend: str
