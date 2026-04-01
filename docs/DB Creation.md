# ML Service – IPL Prediction System

FastAPI microservice with MongoDB integration for IPL match predictions.

**Stack:** FastAPI · MongoDB · Docker · Pydantic

## Structure
```bash
ml-service/
├── app/main.py
├── requirements-prod.txt
└── Dockerfile
```

## Run
```bash
docker compose up --build  # http://localhost:5001
```

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Service status |
| POST | `/predict` | Match prediction |
| GET | `/db-check` | DB connectivity |

## Predict Payload
```json
{ "team_a": "CSK", "team_b": "MI" }
```

## Config
```env
MONGO_URI=mongodb://mongo:27017/ipl_db
```
Supports local Docker Mongo and Atlas. API docs at `/docs`.

> ⚠️ Prediction logic is a placeholder — ready for real model integration.

Errors
-------
Errors Encountered:

- ModuleNotFoundError: No module named 'pymongo'
  → Backend service crashed because required dependency was missing
  
- 404 Not Found for /db-check
  → Docker container was running old code (image not rebuilt)

- SSL handshake failed (MongoDB Atlas)
  → TLS verification issue inside Docker, fixed using certifi
