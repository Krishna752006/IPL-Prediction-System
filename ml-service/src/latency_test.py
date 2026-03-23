import pickle
import time

import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model("../models/lstm1.keras")

# Load preprocessing objects
with open("../models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("../models/scalerx.pkl", "rb") as f:
    scalerx = pickle.load(f)

with open("../models/scalery.pkl", "rb") as f:
    scalery = pickle.load(f)

# ----------- INPUT -----------
# Example input (change values)
input_data = {
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "over": 10,
    "ball": 3,
}

# Convert to model format
cat_input = [
    [input_data["batting_team"], input_data["bowling_team"], input_data["venue"]]
]

num_input = [[input_data["over"], input_data["ball"]]]

# Encode
encoded_cat = encoder.transform(cat_input)
scaled_num = scalerx.transform(num_input)

# Combine
X = np.hstack((encoded_cat, scaled_num))
X = X.reshape(1, 1, X.shape[1])

# ----------- LATENCY MEASURE -----------
start_time = time.time()

y_pred = model.predict(X, verbose=0)

end_time = time.time()

latency_ms = (end_time - start_time) * 1000

# Convert back to real values
y_pred_real = scalery.inverse_transform(y_pred)

# ----------- OUTPUT -----------
runs, wickets = y_pred_real[0]

print(f"Predicted Runs: {runs:.2f}")
print(f"Predicted Wickets: {wickets:.2f}")
print(f"Latency: {latency_ms:.3f} ms")
