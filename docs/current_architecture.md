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
