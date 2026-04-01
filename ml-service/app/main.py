import os

import certifi
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="ML Service", version="1.0")

# -----------------------------
# MongoDB Setup
# -----------------------------
mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/ipl_db")
client = MongoClient(
    mongo_uri, tls=True, tlsCAFile=certifi.where()  # 🔥 THIS FIXES SSL
)
db = client["ipl_db"]


# -----------------------------
# Pydantic Models
# -----------------------------
class PredictRequest(BaseModel):
    team_a: str
    team_b: str


class PredictResponse(BaseModel):
    prediction: str


class DBResponse(BaseModel):
    db: str
    message: str | None = None


# -----------------------------
# Routes
# -----------------------------


# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}


# Dummy Prediction
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Dummy logic
    return {"prediction": f"{request.team_a}_wins"}


# DB Connection Check
@app.get("/db-check", response_model=DBResponse)
def db_check():
    try:
        client.admin.command("ping")
        return {"db": "connected"}
    except Exception as e:
        return {"db": "error", "message": str(e)}
