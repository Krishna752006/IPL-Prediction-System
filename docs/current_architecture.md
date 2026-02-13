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

React → Flask → Encoder → Scaler → LSTM → Prediction  
                 ↓  
          Background retraining

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
