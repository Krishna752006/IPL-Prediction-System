from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from schemas.match import PredictMatchRequest, PredictMatchResponse
from services.match_engine import simulate_match
from services.squads import list_team_codes

logger = logging.getLogger(__name__)

router = APIRouter(tags=["simulation"])


@router.post("/predict-1-match", response_model=PredictMatchResponse)
def predict_one_match(req: PredictMatchRequest) -> PredictMatchResponse:
    valid_codes = set(list_team_codes())
    if req.team_a not in valid_codes or req.team_b not in valid_codes:
        raise HTTPException(
            status_code=404,
            detail=f"team_a/team_b must be one of {sorted(valid_codes)}",
        )
    if req.team_a == req.team_b:
        raise HTTPException(status_code=422, detail="team_a and team_b must differ")
    if req.toss_winner is not None and req.toss_winner not in (req.team_a, req.team_b):
        raise HTTPException(
            status_code=422, detail="toss_winner must equal team_a or team_b"
        )
    if req.toss_decision is not None and req.toss_decision not in ("bat", "bowl"):
        raise HTTPException(
            status_code=422, detail="toss_decision must be 'bat' or 'bowl'"
        )

    try:
        result = simulate_match(
            team_a_code=req.team_a,
            team_b_code=req.team_b,
            venue=req.venue,
            toss_winner=req.toss_winner,
            toss_decision=req.toss_decision,
            seed=req.seed,
        )
    except Exception:
        logger.exception("Match simulation failed for %s vs %s", req.team_a, req.team_b)
        raise HTTPException(status_code=500, detail="Match simulation failed")

    return PredictMatchResponse(**result.to_dict())
