from pydantic import BaseModel


class PredictionRequest(BaseModel):
    team1: str
    team2: str
    overs_left: float
    wickets_left: int
    runs_needed: int


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
