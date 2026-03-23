import time

import numpy as np
from core.model_loader import load_production_model

# -----------------------------
# Load model
# -----------------------------
bundle, metadata = load_production_model()

# -----------------------------
# DUMMY INPUT (69 features, exclude y_runs & y_wickets)
# -----------------------------
dummy_input = np.random.rand(1, 69).astype(np.float32)

# -----------------------------
# LATENCY MEASURE
# -----------------------------
start_time = time.time()
pred = bundle.predict(dummy_input)
end_time = time.time()

latency_ms = (end_time - start_time) * 1000

# -----------------------------
# OUTPUT
# -----------------------------
runs, wickets = pred[0]
print(f"Predicted Runs:    {runs:.2f}")
print(f"Predicted Wickets: {wickets:.2f}")
print(f"Latency:           {latency_ms:.3f} ms")
