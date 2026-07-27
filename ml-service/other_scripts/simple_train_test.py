import argparse
import os
import pickle
import random
import shutil

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from core.config_loader import load_config
from core.logger import setup_logger
from core.model_bundle import IPLModelBundle
from core.registry import promote_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")

args = parser.parse_args()
config = load_config(args.config)

logger = setup_logger(__name__)
logger.info("Training script started")
logger.info(f"Using config: {args.config}")

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SERVICE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(ML_SERVICE_DIR, config["data"]["path"])
STAGING_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["staging_dir"])
PRODUCTION_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["production_dir"])
MLFLOW_DB = os.path.join(ML_SERVICE_DIR, config["paths"]["mlflow_db"])

HISTORY_DIR = os.path.join(ML_SERVICE_DIR, "models", "history")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(PRODUCTION_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# -----------------------------
# CONFIG VALUES
# -----------------------------
EXPERIMENT_NAME = config["experiment"]["name"]

DATASET_VERSION = config["data"]["dataset_version"]
FEATURE_VERSION = config["data"]["feature_version"]
TEST_SIZE = config["data"]["test_size"]
VAL_SIZE = config["data"]["val_size"]
SEED = config["data"]["random_seed"]

EPOCHS = config["model"]["epochs"]
BATCH_SIZE = config["model"]["batch_size"]

GATE_MAE = config["validation_gate"]["test_mae"]
GATE_R2 = config["validation_gate"]["test_r2"]

LAYER1_UNITS = config["model"]["lstm1_units"]
LAYER2_UNITS = config["model"]["lstm2_units"]
DENSE1 = config["model"]["dense1"]
DENSE2 = config["model"]["dense2"]
LEARNING_RATE = config["model"]["learning_rate"]
MODEL_TYPE = config["model"]["type"]


# -----------------------------
# LAYER FACTORY
# -----------------------------
def get_recurrent_layer(model_type, units, return_sequences=False):
    """Return the appropriate recurrent layer based on model_type in config."""
    model_type = model_type.upper()
    if model_type == "LSTM":
        return keras.layers.LSTM(units, return_sequences=return_sequences)
    elif model_type == "GRU":
        return keras.layers.GRU(units, return_sequences=return_sequences)
    elif model_type == "RNN":
        return keras.layers.SimpleRNN(units, return_sequences=return_sequences)
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Choose LSTM, GRU, or RNN."
        )


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# MLFLOW
# -----------------------------
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# LOAD DATA
# -----------------------------
logger.info("Loading dataset")
logger.info(f"Dataset path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found: {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

metadata_path = os.path.join(
    ML_SERVICE_DIR, "data", "metadata", f"{DATASET_VERSION}.json"
)

if not os.path.exists(metadata_path):
    logger.error(f"Metadata missing for dataset {DATASET_VERSION}")
    raise FileNotFoundError(f"Metadata not found for dataset {DATASET_VERSION}")

if DATA_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATA_PATH)
else:
    raise ValueError("Unsupported data format")

# -----------------------------
# FEATURES
# -----------------------------
X = df.drop(columns=["y_runs", "y_wickets"]).values
y = df[["y_runs", "y_wickets"]].values

# reshape for LSTM, RNN, GRU
X = X.reshape(X.shape[0], 1, X.shape[1])

# -----------------------------
# TRAINING
# -----------------------------
with mlflow.start_run():

    try:
        logger.info("MLflow run started")
        logger.info(f"Experiment: {EXPERIMENT_NAME}")

        # -----------------------------
        # LOG PARAMS
        # -----------------------------
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("dataset_path", DATA_PATH)

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("val_size", VAL_SIZE)
        mlflow.log_param("seed", SEED)

        mlflow.log_param("layer1_units", LAYER1_UNITS)
        mlflow.log_param("layer2_units", LAYER2_UNITS)
        mlflow.log_param("dense1", DENSE1)
        mlflow.log_param("dense2", DENSE2)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        mlflow.log_param("config_file", args.config)

        mlflow.log_artifact(metadata_path)

        # -----------------------------
        # SPLIT
        # -----------------------------
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED
        )

        logger.info("Data split completed")
        logger.info(f"Train size: {X_train.shape}")
        logger.info(f"Val size: {X_val.shape}")
        logger.info(f"Test size: {X_test.shape}")

        # -----------------------------
        # MODEL
        # -----------------------------

        logger.info(f"Building {MODEL_TYPE} model")
        logger.info(f"Layer1: {LAYER1_UNITS}, Layer2: {LAYER2_UNITS}")
        logger.info(f"Dense layers: {DENSE1}, {DENSE2}")

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1, X.shape[2])),
                get_recurrent_layer(MODEL_TYPE, LAYER1_UNITS, return_sequences=True),
                get_recurrent_layer(MODEL_TYPE, LAYER2_UNITS, return_sequences=False),
                keras.layers.Dense(DENSE1, activation="relu"),
                keras.layers.Dense(DENSE2, activation="relu"),
                keras.layers.Dense(2),
            ]
        )

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[
                "mae",
                "mse",
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.R2Score(name="r2"),
            ],
        )

        # -----------------------------
        # TRAIN
        # -----------------------------
        logger.info("Training started")

        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=1,
        )

        logger.info("Training completed")
        # -----------------------------
        # SAVE BUNDLE
        # -----------------------------
        bundle = IPLModelBundle(
            model=model,
            dataset_version=DATASET_VERSION,
            feature_version=FEATURE_VERSION,
        )

        run_id = mlflow.active_run().info.run_id[:6]

        model_name = (
            f"{EXPERIMENT_NAME}_"
            f"dataset_{DATASET_VERSION}_"
            f"features_{FEATURE_VERSION}_"
            f"run_{run_id}.pkl"
        )

        save_path = os.path.join(STAGING_DIR, model_name)

        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Model saved to staging: {save_path}")

        mlflow.set_tag("model_file", model_name)
        mlflow.set_tag("model_stage", "staging")

        # -----------------------------
        # TRAIN METRICS
        # -----------------------------
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("train_mae", history.history["mae"][-1])
        mlflow.log_metric("val_mae", history.history["val_mae"][-1])

        # -----------------------------
        # TEST
        # -----------------------------
        test_results = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_metric("test_loss", test_results[0])
        mlflow.log_metric("test_mae", test_results[1])
        mlflow.log_metric("test_mse", test_results[2])
        mlflow.log_metric("test_rmse", test_results[3])
        mlflow.log_metric("test_r2", test_results[4])

        logger.info(f"Test results: {test_results}")

        # -----------------------------
        # R2 Calculation
        # -----------------------------
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        n = X_val.shape[0]
        p = X_val.shape[2]

        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        mlflow.log_metric("r2_real", r2)
        mlflow.log_metric("adjusted_r2", adjusted_r2)

        # -----------------------------
        # LOG TF MODEL
        # -----------------------------
        mlflow.tensorflow.log_model(model, name=f"{MODEL_TYPE}_model")

        mlflow.log_artifact(save_path)
        mlflow.log_artifact(args.config)

        logger.info("Training pipeline completed")

        # -----------------------------
        # FIX MLRUNS LOCATION
        # -----------------------------
        stray_mlruns = os.path.join(ML_SERVICE_DIR, "mlruns")
        target_mlruns = os.path.join(ML_SERVICE_DIR, "experiments", "mlruns")

        if os.path.exists(stray_mlruns):

            if not os.path.exists(target_mlruns):
                shutil.move(stray_mlruns, target_mlruns)

            else:
                for item in os.listdir(stray_mlruns):
                    src = os.path.join(stray_mlruns, item)
                    dst = os.path.join(target_mlruns, item)

                    if not os.path.exists(dst):
                        shutil.move(src, dst)

                shutil.rmtree(stray_mlruns)

        # -----------------------------
        # VALIDATION GATE
        # -----------------------------
        test_mae = test_results[1]
        test_r2 = test_results[4]

        logger.info("Model validation started")
        logger.info(f"MAE: {test_mae}")
        logger.info(f"R2: {test_r2}")

        if test_mae < GATE_MAE and test_r2 > GATE_R2:

            logger.info("Model passed validation gate")
            logger.info("Promoting model to production")

            mlflow.set_tag("model_stage", "production")
            mlflow.set_tag("registry_status", "production")
            mlflow.set_tag("production_model_file", model_name)
            mlflow.set_tag("promoted_run_id", run_id)

            mlflow.log_param("promoted_model", model_name)
            mlflow.log_param("production_dataset", DATASET_VERSION)
            mlflow.log_param("production_feature_version", FEATURE_VERSION)

            promote_model(
                model_name=model_name,
                model_type=MODEL_TYPE,
                dataset_version=DATASET_VERSION,
                feature_version=FEATURE_VERSION,
                run_id=run_id,
            )

            logger.info("Model promotion completed")

        else:
            logger.warning("Model did not pass validation gate — remains in staging")

    except Exception as e:
        mlflow.set_tag("status", "failed")
        logger.exception("Training failed")
        raise
