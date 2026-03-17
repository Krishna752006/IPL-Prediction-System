import numpy as np
import pandas as pd
import math
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


print("Started training...")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("../data/data1.csv", index_col="Unnamed: 0")

input_features = ['batting_team', 'bowling_team', 'venue']
xfeats = ['over', 'ball']
yfeats = ['runs', 'wickets']

# -----------------------------
# ENCODE + SCALE
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[input_features])

scalerx = StandardScaler()
scaled_x = scalerx.fit_transform(data[xfeats])

scalery = StandardScaler()
scaled_y = scalery.fit_transform(data[yfeats])

# Final feature matrix
X = np.hstack((encoded_categorical, scaled_x))
y = scaled_y

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate_model(name, model, X_train, y_train, X_test, y_test, scalery):

    model.fit(X_train, y_train)

    # -----------------------------
    # LATENCY MEASUREMENT
    # -----------------------------
    start_time = time.time()

    y_pred = model.predict(X_test)

    end_time = time.time()

    total_latency_ms = (end_time - start_time) * 1000
    avg_latency_per_sample = total_latency_ms / X_test.shape[0]

    # -----------------------------
    # METRICS
    # -----------------------------
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
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
        "Total_Latency_ms": total_latency_ms,
        "Latency_per_sample_ms": avg_latency_per_sample
    }


# -----------------------------
# MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# -----------------------------
# RUN ALL MODELS
# -----------------------------
results = []

for name, model in models.items():
    result = evaluate_model(
        name, model,
        X_train, y_train,
        X_test, y_test,
        scalery
    )
    results.append(result)

# -----------------------------
# LEADERBOARD
# -----------------------------
results_df = pd.DataFrame(results)

# sort by RMSE (best first)
results_df = results_df.sort_values(by="RMSE")

print("\nMODEL LEADERBOARD:\n")
print(results_df.to_string(index=False))
