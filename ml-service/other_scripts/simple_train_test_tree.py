import argparse
import os
import pickle
import random
import shutil

import mlflow
import numpy as np
import pandas as pd
from core.config_loader import load_config
from core.logger import setup_logger
from core.model_bundle import IPLModelBundle
from core.registry import promote_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")

args = parser.parse_args()
config = load_config(args.config)

logger = setup_logger(__name__)
logger.info("Tree training script started")
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
SEED = config["data"]["random_seed"]

GATE_MAE = config["validation_gate"]["test_mae"]
GATE_R2 = config["validation_gate"]["test_r2"]

MODEL_TYPE = config["model"]["type"]


# -----------------------------
# MODEL FACTORY
# -----------------------------
def get_tree_model(model_type, model_cfg):
    """Return the appropriate tree-based model based on model_type in config."""
    model_type = model_type.upper()

    if model_type == "DECISION_TREE":
        base = DecisionTreeRegressor(
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            random_state=SEED,
        )
        return MultiOutputRegressor(base)

    elif model_type == "RANDOM_FOREST":
        return RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=SEED,
        )

    elif model_type == "XGBOOST":
        base = XGBRegressor(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=SEED,
            verbosity=0,
        )
        return MultiOutputRegressor(base)

    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Choose DECISION_TREE, RANDOM_FOREST, or XGBOOST."
        )


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)

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

# Tree models do not need sequence reshaping — use flat 2D input
# X shape: (n_samples, n_features)

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
        model_cfg = config["model"]

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("dataset_path", DATA_PATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("config_file", args.config)

        # Log all model-specific params from config
        for key, value in model_cfg.items():
            if key != "type":
                mlflow.log_param(key, value)

        mlflow.log_artifact(metadata_path)

        # -----------------------------
        # SPLIT
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )

        logger.info("Data split completed")
        logger.info(f"Train size: {X_train.shape}")
        logger.info(f"Test size: {X_test.shape}")

        # -----------------------------
        # MODEL
        # -----------------------------
        logger.info(f"Building {MODEL_TYPE} model")

        model = get_tree_model(MODEL_TYPE, model_cfg)

        # -----------------------------
        # TRAIN
        # -----------------------------
        logger.info("Training started")

        model.fit(X_train, y_train)

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
        # TEST METRICS
        # -----------------------------
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred_test)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        n = X_test.shape[0]
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("adjusted_r2", adjusted_r2)

        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Test R2:  {test_r2:.4f}")
        logger.info(f"Adjusted R2: {adjusted_r2:.4f}")

        # -----------------------------
        # LOG SKLEARN MODEL
        # -----------------------------
        mlflow.sklearn.log_model(model, name=f"{MODEL_TYPE}_model")

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
        logger.info("Model validation started")
        logger.info(f"MAE: {test_mae:.4f}  (gate < {GATE_MAE})")
        logger.info(f"R2:  {test_r2:.4f}  (gate > {GATE_R2})")

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
