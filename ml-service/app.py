from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "ML service running"}


@app.post("/predict")
def predict(data: dict):
    # dummy prediction
    return {"prediction": "Team A wins", "input": data}
