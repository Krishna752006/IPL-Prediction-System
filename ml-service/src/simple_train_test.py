import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras

data = pd.read_csv("../data/data1.csv", index_col="Unnamed: 0")

input_features = ["batting_team", "bowling_team", "venue"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(data[input_features])

xfeats = ["over", "ball"]
scalerx = StandardScaler()
scaled_x = scalerx.fit_transform(data[xfeats])

yfeats = ["runs", "wickets"]
scalery = StandardScaler()
scaled_y = scalery.fit_transform(data[yfeats])

team = {
    1: "Chennai Super Kings",
    2: "Delhi Capitals",
    3: "Gujarat Titans",
    5: "Kolkata Knight Riders",
    6: "Lucknow Super Giants",
    7: "Mumbai Indians",
    9: "Punjab Kings",
    10: "Rajasthan Royals",
    12: "Royal Challengers Bengaluru",
    13: "Sunrisers Hyderabad",
}
venue = {
    1: "Arun Jaitley Stadium",
    3: "Barsapara Cricket Stadium",
    7: "Dr DY Patil Sports Academy",
    10: "Eden Gardens",
    11: "Ekana Cricket Stadium",
    12: "Feroz Shah Kotla",
    18: "M Chinnaswamy Stadium",
    19: "MA Chidambaram Stadium",
    20: "Maharaja Yadavindra Singh International Cricket Stadium",
    22: "Narendra Modi Stadium",
    27: "Punjab Cricket Association IS Bindra Stadium",
    28: "Punjab Cricket Association Stadium",
    29: "Rajiv Gandhi International Stadium",
    30: "Sardar Patel Stadium",
    31: "Saurashtra Cricket Association Stadium",
    32: "Sawai Mansingh Stadium",
    35: "Sheikh Zayed Stadium",
    37: "Subrata Roy Sahara Stadium",
    39: "Vidarbha Cricket Association Stadium",
    40: "Wankhede Stadium",
}
steam = {str(key): value for key, value in team.items()}
svenue = {str(key): value for key, value in venue.items()}

try:
    print("Started background training...")

    encoded_categorical = encoder.fit_transform(data[input_features])
    scaled_x = scalerx.fit_transform(data[xfeats])
    scaled_y = scalery.fit_transform(data[yfeats])

    with open("../models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("../models/scalerx.pkl", "wb") as f:
        pickle.dump(scalerx, f)

    with open("../models/scalery.pkl", "wb") as f:
        pickle.dump(scalery, f)

    X = np.hstack((encoded_categorical, scaled_x))
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = np.vstack((scaled_y))

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(2),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",  # this is MSE
        metrics=[
            "mae",
            "mse",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.R2Score(name="r2"),
        ],
    )

    model.fit(
        X_train, y_train, epochs=10, batch_size=3887, validation_data=(X_val, y_val)
    )

    test_results = model.evaluate(X_test, y_test)

    print(test_results)  # [loss, mae, mse, rmse, r2]

    # Predict
    y_pred = model.predict(X_val)

    # Convert back to real cricket values
    y_pred_real = scalery.inverse_transform(y_pred)
    y_val_real = scalery.inverse_transform(y_val)

    # ✅ R2 on REAL values
    r2 = r2_score(y_val_real, y_pred_real)

    # samples and features
    n = X_val.shape[0]  # number of rows
    p = X_val.shape[2]  # number of input features

    # ✅ Adjusted R2
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print("R2 Score:", r2)
    print("Adjusted R2 Score:", adjusted_r2)


except Exception as e:
    print(f"Error during background training: {e}")
