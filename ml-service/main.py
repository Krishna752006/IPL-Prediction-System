from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictionRequest(BaseModel):
    team1: str
    team2: str
    runs_needed: int


class PredictionResponse(BaseModel):
    prediction: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    return {"prediction": data.team1, "probability": 0.65}
