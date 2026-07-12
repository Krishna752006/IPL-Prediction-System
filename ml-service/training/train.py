import argparse
import json
import os
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
logger.info("TensorFlow Training script started using Pre-trained Embeddings")
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

# Resolve path to your custom Kaggle-generated embeddings JSON
EMBEDDINGS_DIR = os.path.join(ML_SERVICE_DIR, config["paths"].get("embeddings_dir", "saved_seasons"))

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
# HIGH-SPEED EMBEDDING PREPROCESSING LAYER
# -----------------------------
def transform_features_with_temporal_embeddings(df, embeddings_dir):
    """Maps raw categorical fields dynamically year-by-year to prevent data leakage."""
    
    logger.info(f"Dynamically loading temporal embeddings from {embeddings_dir} to prevent data leakage...")
    
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory missing: {embeddings_dir}")

    # Default dimensions established from your TabTransformer config
    p_dim, v_dim, s_dim, m_dim = 60, 30, 8, 8
    n_rows = len(df)
    
    # Pre-allocate zero matrices to maintain exact row ordering
    b_vectors = np.zeros((n_rows, p_dim))
    ns_vectors = np.zeros((n_rows, p_dim))
    bw_vectors = np.zeros((n_rows, p_dim))
    v_vectors = np.zeros((n_rows, v_dim))
    s_vectors = np.zeros((n_rows, s_dim))
    m_vectors = np.zeros((n_rows, m_dim))
    
    # Iterate over each unique season in the dataset
    for season in sorted(df["season"].dropna().unique()):
        # The PyTorch script saves embeddings trained on data *prior* to target_year
        # So to predict season X, we load static_embeddings_X.json
        json_filename = f"static_embeddings_{int(season)}.json"
        json_path = os.path.join(embeddings_dir, json_filename)
        
        # Create a boolean mask for rows belonging to this specific season
        mask = (df["season"] == season).values
        season_df = df[mask]
        
        if not os.path.exists(json_path):
            logger.warning(f"Embeddings for season {int(season)} not found at {json_path}. Using zero vectors.")
            continue
            
        logger.info(f"Applying {json_filename} to {mask.sum()} rows from the {int(season)} season.")
        
        with open(json_path, "r") as f:
            embeds = json.load(f)
            
        # Fast mapping structures for this specific year
        batter_map = {k: v["batter_embedding"] for k, v in embeds.get("players", {}).items()}
        non_striker_map = {k: v["non_striker_embedding"] for k, v in embeds.get("players", {}).items()}
        bowler_map = {k: v["bowler_embedding"] for k, v in embeds.get("players", {}).items()}
        venue_map = embeds.get("venues", {})
        season_map = embeds.get("season", {})
        match_state_list = embeds.get("match_state", [])
        
        def fetch_vec(mapping, keys, default_dim):
            return np.array([mapping.get(str(k), [0.0]*default_dim) for k in keys])
            
        # Fill the allocated matrices only at the indices for this season
        b_vectors[mask] = fetch_vec(batter_map, season_df.get("batsman", [None]*len(season_df)), p_dim)
        ns_vectors[mask] = fetch_vec(non_striker_map, season_df.get("non_striker", [None]*len(season_df)), p_dim)
        bw_vectors[mask] = fetch_vec(bowler_map, season_df.get("bowler", [None]*len(season_df)), p_dim)
        v_vectors[mask] = fetch_vec(venue_map, season_df.get("venue", [None]*len(season_df)), v_dim)
        s_vectors[mask] = fetch_vec(season_map, season_df.get("season", [None]*len(season_df)), s_dim)
        
        if "match_state_id" in season_df.columns:
            m_vectors[mask] = np.array([
                match_state_list[int(x)] if int(x) < len(match_state_list) else [0.0]*m_dim 
                for x in season_df["match_state_id"]
            ])

    unwanted = ["current_score", "is_wicket_target", "isWide_target", "batsman", "non_striker", "bowler", "venue", "season", "match_state_id"]
    drops = [col for col in unwanted if col in df.columns]
    
    numerical_matrix = df.drop(columns=drops).select_dtypes(include=[np.number]).values

    # Glue spatial layers together into unified training payload
    X_engineered = np.hstack([
        numerical_matrix,
        b_vectors,
        ns_vectors,
        bw_vectors,
        v_vectors,
        s_vectors,
        m_vectors
    ])
    
    return X_engineered

# -----------------------------
# LOAD DATA
# -----------------------------
logger.info("Loading dataset")
logger.info(f"Dataset path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found: {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if DATA_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATA_PATH)
else:
    raise ValueError("Unsupported data format")

# -----------------------------
# EXTRACT TARGETS & INJECT EMBEDDINGS
# -----------------------------
y = df[["current_score", "is_wicket_target", "isWide_target"]].values
X = transform_features_with_temporal_embeddings(df, EMBEDDINGS_DIR)

# Reshape clean feature dimensions directly for recurrent engine input topologies
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
        mlflow.log_artifact(EMBEDDINGS_DIR) # Track the underlying geometry version

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
        # MODEL BUILDING
        # -----------------------------
        logger.info(f"Building legacy light-weight structural {MODEL_TYPE} architecture")

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1, X.shape[2])),
                get_recurrent_layer(MODEL_TYPE, LAYER1_UNITS, return_sequences=True),
                get_recurrent_layer(MODEL_TYPE, LAYER2_UNITS, return_sequences=False),
                keras.layers.Dense(DENSE1, activation="relu"),
                keras.layers.Dense(DENSE2, activation="relu"),
                keras.layers.Dense(3),
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
        logger.info("High-speed network training initiated...")

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
            import pickle
            pickle.dump(bundle, f)

        logger.info(f"Model saved to staging area: {save_path}")

        mlflow.set_tag("model_file", model_name)
        mlflow.set_tag("model_stage", "staging")

        # -----------------------------
        # TRAIN METRICS LOGGING
        # -----------------------------
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("train_mae", history.history["mae"][-1])
        mlflow.log_metric("val_mae", history.history["val_mae"][-1])

        # -----------------------------
        # TEST RUN
        # -----------------------------
        test_results = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_metric("test_loss", test_results[0])
        mlflow.log_metric("test_mae", test_results[1])
        mlflow.log_metric("test_mse", test_results[2])
        mlflow.log_metric("test_rmse", test_results[3])
        mlflow.log_metric("test_r2", test_results[4])

        logger.info(f"Test results evaluation context: {test_results}")

        # -----------------------------
        # METRIC CALCULATIONS
        # -----------------------------
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        n = X_val.shape[0]
        p = X_val.shape[2]

        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        mlflow.log_metric("r2_real", r2)
        mlflow.log_metric("adjusted_r2", adjusted_r2)

        # -----------------------------
        # SAVE MLFLOW TARGETS
        # -----------------------------
        mlflow.tensorflow.log_model(model, name=f"{MODEL_TYPE}_model")
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(args.config)

        logger.info("Training pipeline loop completed cleanly")

        # -----------------------------
        # STRAY DIRECTORY CLEANUP
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
        # VALIDATION GATE EVALUATION
        # -----------------------------
        test_mae = test_results[1]
        test_r2 = test_results[4]

        logger.info("MLOps Gate Assessment validation check started")
        logger.info(f"Current Metrics -> MAE: {test_mae} | R2: {test_r2}")

        if test_mae < GATE_MAE and test_r2 > GATE_R2:
            logger.info("✅ Model strictly cleared target metrics gate verification.")
            logger.info("Promoting architectural deployment package to Production registry...")

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
            logger.info("Model verification promotion phase completed.")
        else:
            logger.warning("⚠ Model parameters missed gate margins — deployment package preserved in Staging registry.")

    except Exception as e:
        mlflow.set_tag("status", "failed")
        logger.exception("MLOps script training thread failure encountered")
        raise