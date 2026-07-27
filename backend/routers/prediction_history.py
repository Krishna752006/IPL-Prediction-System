from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from datetime import datetime

from database import prediction_history_collection, users_collection
from security import verify_token

router = APIRouter()
security = HTTPBearer()

class PredictionHistoryCreate(BaseModel):
    team_a: str
    team_b: str
    venue: str
    winner: str
    result: str


class PredictionHistoryResponse(BaseModel):
    id: str
    team_a: str
    team_b: str
    venue: str
    winner: str
    result: str
    predicted_at: datetime

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    email = payload["sub"]

    user = users_collection.find_one(
        {"email": email}
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user

@router.post("/prediction-history")
def save_prediction(
    prediction: PredictionHistoryCreate,
    user=Depends(get_current_user)
):
        prediction_history_collection.insert_one({

        "user_id": str(user["_id"]),

        "team_a": prediction.team_a,

        "team_b": prediction.team_b,

        "venue": prediction.venue,

        "winner": prediction.winner,

        "result": prediction.result,

        "predicted_at": datetime.utcnow()

    })

        return {
            "success": True,
            "message": "Prediction saved."
        }

@router.get("/prediction-history")
def get_prediction_history(
    user=Depends(get_current_user)
):
    predictions = prediction_history_collection.find(
    {
        "user_id": str(user["_id"])
    }
    ).sort(
        "predicted_at",
        -1
    )

    history = []

    for prediction in predictions:

        history.append({

            "id": str(prediction["_id"]),

            "team_a": prediction["team_a"],

            "team_b": prediction["team_b"],

            "venue": prediction["venue"],

            "winner": prediction["winner"],

            "result": prediction["result"],

            "predicted_at": prediction["predicted_at"]

        })

    return history