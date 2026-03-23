import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tensorflow import keras
from xgboost import XGBRegressor

print("Started training baselines...")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("../data/data1.csv", index_col="Unnamed: 0")

input_features = ["batting_team", "bowling_team", "venue"]
xfeats = ["over", "ball"]
yfeats = ["runs", "wickets"]

# -----------------------------
# PREPROCESS (SELF-CONTAINED)
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scalerx = StandardScaler()
scalery = StandardScaler()

encoded_categorical = encoder.fit_transform(data[input_features])
scaled_x = scalerx.fit_transform(data[xfeats])
scaled_y = scalery.fit_transform(data[yfeats])

X = np.hstack((encoded_categorical, scaled_x))
y = scaled_y

# -----------------------------
# SPLIT
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


# -----------------------------
# EVALUATION FUNCTION (sklearn)
# -----------------------------
def evaluate_model(name, model):

    model.fit(X_train, y_train)

    # latency
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    total_latency_ms = (end_time - start_time) * 1000
    avg_latency = total_latency_ms / X_test.shape[0]

    # metrics (REAL SCALE)
    y_pred_real = scalery.inverse_transform(y_pred)
    y_test_real = scalery.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred_real)

    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adj_R2": adjusted_r2,
        "Latency(ms)": total_latency_ms,
        "Per_Sample(ms)": avg_latency,
    }


# -----------------------------
# RECURRENT MODEL CONFIG
# -----------------------------
LAYER1_UNITS = 100
LAYER2_UNITS = 50
DENSE1 = 10
DENSE2 = 10

# Reshape X for recurrent models: (samples, timesteps=1, features)
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


def get_recurrent_layer(model_type, units, return_sequences):
    """Return the appropriate Keras recurrent layer."""
    if model_type == "LSTM":
        return keras.layers.LSTM(units, return_sequences=return_sequences)
    elif model_type == "GRU":
        return keras.layers.GRU(units, return_sequences=return_sequences)
    elif model_type == "RNN":
        return keras.layers.SimpleRNN(units, return_sequences=return_sequences)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_recurrent_model(model_type):
    """Build a two-layer recurrent model followed by dense layers."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, X_train.shape[1])),
            get_recurrent_layer(model_type, LAYER1_UNITS, return_sequences=True),
            get_recurrent_layer(model_type, LAYER2_UNITS, return_sequences=False),
            keras.layers.Dense(DENSE1, activation="relu"),
            keras.layers.Dense(DENSE2, activation="relu"),
            keras.layers.Dense(2),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_recurrent_model(model_type):
    """Train and evaluate a recurrent model, returning a metrics dict."""
    print(f"  Training {model_type}...")
    model = build_recurrent_model(model_type)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=10,
        batch_size=1048,
        callbacks=[early_stop],
        verbose=0,
    )

    # latency
    start_time = time.time()
    y_pred = model.predict(X_test_seq, verbose=0)
    end_time = time.time()

    total_latency_ms = (end_time - start_time) * 1000
    avg_latency = total_latency_ms / X_test_seq.shape[0]

    # metrics (REAL SCALE)
    y_pred_real = scalery.inverse_transform(y_pred)
    y_test_real = scalery.inverse_transform(y_test)

    mse = mean_squared_error(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred_real)

    n = X_test_seq.shape[0]
    p = X_test_seq.shape[1] * X_test_seq.shape[2]  # flattened feature count
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "Model": model_type,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adj_R2": adjusted_r2,
        "Latency(ms)": total_latency_ms,
        "Per_Sample(ms)": avg_latency,
    }


# -----------------------------
# SKLEARN MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=20, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, model in models.items():
    results.append(evaluate_model(name, model))

# -----------------------------
# RECURRENT MODELS
# -----------------------------
for rnn_type in ["LSTM", "GRU", "RNN"]:
    results.append(evaluate_recurrent_model(rnn_type))

# -----------------------------
# LEADERBOARD
# -----------------------------
results_df = pd.DataFrame(results).sort_values(by="RMSE")

print("\nMODEL LEADERBOARD:\n")
print(results_df.to_string(index=False))
