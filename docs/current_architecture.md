# Current Architecture

Frontend (React)
     ↓
Flask API
     ↓
Loaded LSTM model
     ↓
Prediction returned

Post-prediction:
- Synthetic rows appended to dataset
- Background thread retrains model
- Model overwritten

Frontend → localhost:5173  
Backend → localhost:5000  

## Performance Baseline

Current prediction latency: ~2.7 seconds

Observed via browser network tab during local testing.

Notes:
- Latency includes preprocessing + model inference.
- Background retraining is triggered asynchronously after response.
- No optimization has been applied yet.

Target latency (future): < 1 second

## Known Problems
- Training occurs inside API
- No model versioning
- Encoder refitted each training cycle
- No experiment tracking

## Planned Architectural Shift

The current system tightly couples model training with the prediction API.

Future iterations will decouple these layers to ensure:

- Faster inference
- Safer model updates
- Improved system reliability
- Production readiness

## Frontend Routes (GET)

:5173/ → Home  
:5173/login  
:5173/register  
:5173/predictions  
:5173/statistics  
:5173/team-analysis

## Backend Endpoint

POST :5000/api/predict

Consumes:
- teamA
- teamB
- venue

Returns:
- predicted score
- wickets
- run_rate
- match_result

## Not implemented.

System currently operates stateless:
- No database
- No session storage
- No user persistence

Login/Register exist only at UI level.
No real authentication mechanism is implemented.
