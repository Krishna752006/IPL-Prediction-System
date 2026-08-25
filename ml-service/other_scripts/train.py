import argparse
import json
import math
import os
import random
import shutil

# from tqdm.keras import TqdmCallback
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from core.config_loader import load_config
from core.logger import setup_logger
from core.model_bundle import IPLModelBundle
from core.registry import promote_model
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
)
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")

args = parser.parse_args()
config = load_config(args.config)

logger = setup_logger(__name__)
logger.info("TensorFlow Training script started using Pre-trained Embeddings")
logger.info(f"Using config: {args.config}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SERVICE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(ML_SERVICE_DIR, config["data"]["path"])
STAGING_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["staging_dir"])
PRODUCTION_DIR = os.path.join(ML_SERVICE_DIR, config["paths"]["production_dir"])
MLFLOW_DB = os.path.join(ML_SERVICE_DIR, config["paths"]["mlflow_db"])
HISTORY_DIR = os.path.join(ML_SERVICE_DIR, "models", "history")

EMBEDDINGS_DIR = os.path.join(
    ML_SERVICE_DIR, config["paths"].get("embeddings_json", "saved_seasons")
)

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(PRODUCTION_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

EXPERIMENT_NAME = config["experiment"]["name"]

DATASET_VERSION = config["data"]["dataset_version"]
FEATURE_VERSION = config["data"]["feature_version"]
SEED = config["data"]["random_seed"]

EPOCHS = config["model"]["epochs"]
BATCH_SIZE = config["model"]["batch_size"]

GATE_MAE = config["validation_gate"]["test_mae"]
GATE_MSE = config["validation_gate"]["test_mse"]
GATE_RMSE = config["validation_gate"]["test_rmse"]
GATE_R2 = config["validation_gate"]["test_r2"]

LAYER1_UNITS = config["model"]["lstm1_units"]
LAYER2_UNITS = config["model"]["lstm2_units"]
DENSE1 = config["model"]["dense1"]
DENSE2 = config["model"]["dense2"]
LEARNING_RATE = config["model"]["learning_rate"]
MODEL_TYPE = "LSTM"

LSTM_DROPOUT = config["model"].get("lstm_dropout", 0.0)
LSTM_RECURRENT_DROPOUT = config["model"].get("lstm_recurrent_dropout", 0.0)

feature_columns = [
    "inning",
    "over",
    "total_balls",
    "balls_remaining",
    "phase_pp",
    "phase_middle",
    "phase_death",
    "target",
    "is_pacer",
    "wickets_before",
    "percentage_target_achieved",
    "current_run_rate",
    "required_run_rate",
    "sin_ball",
    "cos_ball",
    "rr_momentum",
    "toss_won",
    "venue_phase_avg",
    "batter_history_matches",
    "last_1_runs",
    "last_1_balls",
    "last_2_runs",
    "last_2_balls",
    "last_3_runs",
    "last_3_balls",
    "bowler_history_matches",
    "last_1_runs_conceded",
    "last_1_balls_bowled",
    "last_2_runs_conceded",
    "last_2_balls_bowled",
    "last_3_runs_conceded",
    "last_3_balls_bowled",
]


def get_lstm_layer(units, return_sequences=False):
    return keras.layers.LSTM(
        units=units,
        return_sequences=return_sequences,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
    )


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)

SCORE_SCALE = 180.0
WICKET_THRESH = config["model"].get("wicket_threshold", 0.5)
WIDE_THRESH = config["model"].get("wide_threshold", 0.5)


def scaled_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square((y_true * SCORE_SCALE) - (y_pred * SCORE_SCALE)))


def scaled_rmse(y_true, y_pred):
    return tf.sqrt(scaled_mse(y_true, y_pred))


def scaled_r2(y_true, y_pred):
    y_t = y_true * SCORE_SCALE
    y_p = y_pred * SCORE_SCALE
    ss_res = tf.reduce_sum(tf.square(y_t - y_p))
    ss_tot = tf.reduce_sum(tf.square(y_t - tf.reduce_mean(y_t)))
    return 1.0 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))


def wicket_mcc(y_true, y_pred):
    y_t = tf.cast(y_true, tf.float32)
    # Apply sigmoid since the loss uses from_logits=True
    y_p = tf.cast(tf.math.sigmoid(y_pred) > WICKET_THRESH, tf.float32)

    tp = tf.reduce_sum(y_t * y_p)
    tn = tf.reduce_sum((1.0 - y_t) * (1.0 - y_p))
    fp = tf.reduce_sum((1.0 - y_t) * y_p)
    fn = tf.reduce_sum(y_t * (1.0 - y_p))

    num = (tp * tn) - (fp * fn)
    den = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return tf.math.divide_no_nan(num, den)


def wide_mcc(y_true, y_pred):
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(tf.math.sigmoid(y_pred) > WIDE_THRESH, tf.float32)

    tp = tf.reduce_sum(y_t * y_p)
    tn = tf.reduce_sum((1.0 - y_t) * (1.0 - y_p))
    fp = tf.reduce_sum((1.0 - y_t) * y_p)
    fn = tf.reduce_sum(y_t * (1.0 - y_p))

    num = (tp * tn) - (fp * fn)
    den = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return tf.math.divide_no_nan(num, den)


class BatchMetricsPrinter(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Pull batch metrics
        score_mse = logs.get("score_scaled_mse", 0.0)
        score_rmse = logs.get("score_scaled_rmse", 0.0)
        score_r2 = logs.get("score_scaled_r2", 0.0)
        wicket_mcc_val = logs.get("wicket_wicket_mcc", 0.0)
        wide_mcc_val = logs.get("wide_wide_mcc", 0.0)

        # Print directly to terminal on a single uncluttered line
        print(
            f"Batch {batch + 1:03d} | "
            f"Score MSE: {score_mse:.1f} | "
            f"RMSE: {score_rmse:.1f} | "
            f"R2: {score_r2:.3f} | "
            f"Wicket MCC: {wicket_mcc_val:.3f} | "
            f"Wide MCC: {wide_mcc_val:.3f}"
        )


def transform_features_with_temporal_embeddings(df, embeddings_dir):
    logger.info(
        f"Dynamically loading temporal embeddings from {embeddings_dir} to prevent data leakage..."
    )

    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory missing: {embeddings_dir}")

    p_dim, v_dim, s_dim, m_dim = 60, 30, 8, 8
    n_rows = len(df)

    b_vectors = np.zeros((n_rows, p_dim), dtype=np.float32)
    ns_vectors = np.zeros((n_rows, p_dim), dtype=np.float32)
    bw_vectors = np.zeros((n_rows, p_dim), dtype=np.float32)
    v_vectors = np.zeros((n_rows, v_dim), dtype=np.float32)
    s_vectors = np.zeros((n_rows, s_dim), dtype=np.float32)
    m_vectors = np.zeros((n_rows, m_dim), dtype=np.float32)

    for season in sorted(df["season"].dropna().unique()):
        json_filename = f"static_embeddings_{int(season)}.json"
        json_path = os.path.join(embeddings_dir, json_filename)

        mask = (df["season"] == season).values
        season_df = df[mask]

        if not os.path.exists(json_path):
            logger.warning(
                f"Embeddings for season {int(season)} not found at {json_path}. Using zero vectors."
            )
            continue

        logger.info(
            f"Applying {json_filename} to {mask.sum()} rows from the {int(season)} season."
        )

        with open(json_path, "r") as f:
            embeds = json.load(f)

        batter_map = {
            k: v["batter_embedding"] for k, v in embeds.get("players", {}).items()
        }
        non_striker_map = {
            k: v["non_striker_embedding"] for k, v in embeds.get("players", {}).items()
        }
        bowler_map = {
            k: v["bowler_embedding"] for k, v in embeds.get("players", {}).items()
        }
        venue_map = embeds.get("venues", {})
        season_map = embeds.get("season", {})
        match_state_list = embeds.get("match_state", [])

        def fetch_vec(mapping, keys, default_dim):
            return np.array([mapping.get(str(k), [0.0] * default_dim) for k in keys])

        b_vectors[mask] = fetch_vec(
            batter_map, season_df.get("batsman", [None] * len(season_df)), p_dim
        )
        ns_vectors[mask] = fetch_vec(
            non_striker_map,
            season_df.get("non_striker", [None] * len(season_df)),
            p_dim,
        )
        bw_vectors[mask] = fetch_vec(
            bowler_map, season_df.get("bowler", [None] * len(season_df)), p_dim
        )
        v_vectors[mask] = fetch_vec(
            venue_map, season_df.get("venue", [None] * len(season_df)), v_dim
        )
        s_vectors[mask] = fetch_vec(
            season_map,
            [
                int(float(x)) if pd.notna(x) else None
                for x in season_df.get("season", [None] * len(season_df))
            ],
            s_dim,
        )

        if "match_state_id" in season_df.columns:
            m_vectors[mask] = np.array(
                [
                    (
                        match_state_list[int(float(x))]
                        if pd.notna(x) and int(float(x)) < len(match_state_list)
                        else [0.0] * m_dim
                    )
                    for x in season_df["match_state_id"]
                ]
            )

    valid_features = [col for col in feature_columns if col in df.columns]
    numerical_matrix = df[valid_features].fillna(0).values.astype(np.float32)

    X_engineered = np.hstack(
        [
            numerical_matrix,
            b_vectors,
            ns_vectors,
            bw_vectors,
            v_vectors,
            s_vectors,
            m_vectors,
        ]
    )

    return X_engineered


logger.info("Loading dataset")
logger.info(f"Dataset path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found: {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if DATA_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATA_PATH)
    df = df[df["season"] >= 2010].reset_index(drop=True)
else:
    raise ValueError("Unsupported data format")

X_raw = transform_features_with_temporal_embeddings(df, EMBEDDINGS_DIR)
y_raw = df[["current_score", "is_wicket_target", "isWide_target"]].values

SEQ_LEN = 30

seq_starts = np.zeros(len(df), dtype=int)
grouped = df.groupby(["matchId", "inning"])
for _, group in grouped:
    group_indices = group.index.values
    group_start = group_indices[0]
    for i in group_indices:
        seq_starts[i] = max(group_start, i - SEQ_LEN + 1)

train_mask = (df["season"] >= 2010) & (df["season"] <= 2024)
train_idx = df.index[train_mask].to_numpy()

df_2025 = df[df["season"] == 2025]

matches_2025 = df_2025["matchId"].unique()

midpoint = len(matches_2025) // 2

val_matches = matches_2025[:midpoint]
test_matches = matches_2025[midpoint:]

val_idx = df_2025[df_2025["matchId"].isin(val_matches)].index.to_numpy()
test_idx = df_2025[df_2025["matchId"].isin(test_matches)].index.to_numpy()

y_train = {
    "score": y_raw[train_idx, 0],
    "wicket": y_raw[train_idx, 1],
    "wide": y_raw[train_idx, 2],
}
y_val = {
    "score": y_raw[val_idx, 0],
    "wicket": y_raw[val_idx, 1],
    "wide": y_raw[val_idx, 2],
}
y_test = {
    "score": y_raw[test_idx, 0],
    "wicket": y_raw[test_idx, 1],
    "wide": y_raw[test_idx, 2],
}

logger.info("Data split completed via indices")
logger.info(f"Train size: {len(train_idx)}")
logger.info(f"Val size: {len(val_idx)}")
logger.info(f"Test size: {len(test_idx)}")


class IPLDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_raw,
        y_raw,
        seq_starts,
        indices,
        batch_size,
        seq_len=30,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.seq_starts = seq_starts
        self.indices = indices
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        X_batch = np.zeros(
            (len(batch_indices), self.seq_len, self.X_raw.shape[1]), dtype=np.float32
        )
        y_score = np.zeros(len(batch_indices), dtype=np.float32)
        y_wicket = np.zeros(len(batch_indices), dtype=np.float32)
        y_wide = np.zeros(len(batch_indices), dtype=np.float32)

        for b_idx, i in enumerate(batch_indices):
            start = self.seq_starts[i]
            seq = self.X_raw[start : i + 1]

            if len(seq) < self.seq_len:
                pad = np.zeros(
                    (self.seq_len - len(seq), self.X_raw.shape[1]), dtype=np.float32
                )
                seq = np.vstack([pad, seq])

            X_batch[b_idx] = seq
            y_score[b_idx] = self.y_raw[i, 0]
            y_wicket[b_idx] = self.y_raw[i, 1]
            y_wide[b_idx] = self.y_raw[i, 2]

        return X_batch, {"score": y_score, "wicket": y_wicket, "wide": y_wide}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


train_gen = IPLDataGenerator(
    X_raw, y_raw, seq_starts, train_idx, BATCH_SIZE, SEQ_LEN, shuffle=True
)
val_gen = IPLDataGenerator(
    X_raw, y_raw, seq_starts, val_idx, BATCH_SIZE, SEQ_LEN, shuffle=False
)
test_gen = IPLDataGenerator(
    X_raw, y_raw, seq_starts, test_idx, BATCH_SIZE, SEQ_LEN, shuffle=False
)

with mlflow.start_run():
    try:
        logger.info("MLflow run started")
        logger.info(f"Experiment: {EXPERIMENT_NAME}")

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("dataset_path", DATA_PATH)

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("test_size", len(test_idx))
        mlflow.log_param("val_size", len(val_idx))
        mlflow.log_param("seed", SEED)

        mlflow.log_param("layer1_units", LAYER1_UNITS)
        mlflow.log_param("layer2_units", LAYER2_UNITS)
        mlflow.log_param("dense1", DENSE1)
        mlflow.log_param("dense2", DENSE2)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        mlflow.log_param("lstm_dropout", LSTM_DROPOUT)
        mlflow.log_param("lstm_recurrent_dropout", LSTM_RECURRENT_DROPOUT)

        mlflow.log_param("config_file", args.config)
        mlflow.log_artifact(EMBEDDINGS_DIR)

        logger.info(
            f"Building legacy light-weight structural {MODEL_TYPE} architecture"
        )

        num_numerical_features = len(feature_columns)
        inputs = keras.layers.Input(shape=(SEQ_LEN, X_raw.shape[1]))
        numerical = inputs[:, :, :num_numerical_features]
        categorical = inputs[:, :, num_numerical_features:]

        numerical_normed = keras.layers.LayerNormalization()(numerical)
        combined = keras.layers.Concatenate()([numerical_normed, categorical])

        x = get_lstm_layer(LAYER1_UNITS, return_sequences=True)(combined)
        lstm_out = get_lstm_layer(LAYER2_UNITS, return_sequences=False)(x)

        shared_mlp = keras.layers.Dense(DENSE1, activation="relu")(lstm_out)
        shared_mlp = keras.layers.Dropout(0.2)(shared_mlp)
        shared_mlp = keras.layers.Dense(DENSE2, activation="relu")(shared_mlp)

        wicket_lstm_out = get_lstm_layer(64, return_sequences=False)(combined)

        score_head1 = keras.layers.Dense(128, activation="relu")(shared_mlp)
        score_head1 = keras.layers.Dropout(0.2)(score_head1)
        score_head2 = keras.layers.Dense(64, activation="relu")(score_head1)
        score_head2 = keras.layers.Dropout(0.2)(score_head2)
        score_head3 = keras.layers.Dense(32, activation="relu")(score_head2)
        score_head3 = keras.layers.Dropout(0.2)(score_head3)
        score_output = keras.layers.Dense(1, name="score")(score_head3)

        wicket_head1 = keras.layers.Dense(64, activation="relu")(wicket_lstm_out)
        wicket_head1 = keras.layers.Dropout(0.2)(wicket_head1)
        wicket_head2 = keras.layers.Dense(32, activation="relu")(wicket_head1)
        wicket_head2 = keras.layers.Dropout(0.2)(wicket_head2)
        wicket_head3 = keras.layers.Dense(16, activation="relu")(wicket_head2)
        wicket_head3 = keras.layers.Dropout(0.2)(wicket_head3)
        wicket_output = keras.layers.Dense(1, name="wicket")(wicket_head3)

        wide_head1 = keras.layers.Dense(48, activation="relu")(shared_mlp)
        wide_head1 = keras.layers.Dropout(0.2)(wide_head1)
        wide_head2 = keras.layers.Dense(16, activation="relu")(wide_head1)
        wide_head2 = keras.layers.Dropout(0.2)(wide_head2)
        wide_output = keras.layers.Dense(1, name="wide")(wide_head2)

        model = keras.Model(
            inputs=inputs, outputs=[score_output, wicket_output, wide_output]
        )

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss={
                "score": tf.keras.losses.Huber(delta=0.14),
                "wicket": tf.keras.losses.BinaryCrossentropy(from_logits=True),
                "wide": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            },
            loss_weights={"score": 2.0, "wicket": 1.0, "wide": 0.8},
            metrics={
                "score": [scaled_mse, scaled_rmse, scaled_r2],
                "wicket": [wicket_mcc],
                "wide": [wide_mcc],
            },
        )

        logger.info("High-speed network training initiated...")

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            verbose=0,
            callbacks=[BatchMetricsPrinter()],
        )

        logger.info("Training completed")

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

        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("train_mae", history.history["score_mae"][-1])
        mlflow.log_metric("val_mae", history.history["val_score_mae"][-1])

        test_results = model.evaluate(test_gen, verbose=0, return_dict=True)

        logger.info(f"Test results evaluation context: {test_results}")
        y_test_pred = model.predict(test_gen)

        y_val_pred = model.predict(val_gen)
        val_score_preds = y_val_pred[0].flatten()
        val_r2_val = r2_score(y_val["score"], val_score_preds)

        n, p = len(val_idx), X_raw.shape[1]
        adjusted_r2 = 1 - (1 - val_r2_val) * (n - 1) / (n - p - 1)
        mlflow.log_metric("val_r2_real", val_r2_val)
        mlflow.log_metric("val_adjusted_r2", adjusted_r2)

        test_score_preds = y_test_pred[0].flatten() * 180
        y_test_score_scaled = y_test["score"] * 180

        test_r2_val = r2_score(y_test_score_scaled, test_score_preds)
        test_n, test_p = len(test_idx), X_raw.shape[1]
        test_adjusted_r2 = 1 - (1 - test_r2_val) * (test_n - 1) / (test_n - test_p - 1)
        mae = np.mean(np.abs(test_score_preds - y_test_score_scaled))
        mse = np.mean((test_score_preds - y_test_score_scaled) ** 2)
        rmse = np.sqrt(mse)

        mean_true, mean_pred = np.mean(y_test_score_scaled), np.mean(test_score_preds)
        var_true, var_pred = np.var(y_test_score_scaled), np.var(test_score_preds)
        cov = np.mean(
            (y_test_score_scaled - mean_true) * (test_score_preds - mean_pred)
        )
        ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

        bias = np.mean(test_score_preds - y_test_score_scaled)
        within_10 = np.mean(np.abs(test_score_preds - y_test_score_scaled) <= 10) * 100
        within_20 = np.mean(np.abs(test_score_preds - y_test_score_scaled) <= 20) * 100

        mlflow.log_metric("test_loss", test_results["loss"])
        mlflow.log_metric("test_mae_runs", mae)
        mlflow.log_metric("test_mse_runs", mse)
        mlflow.log_metric("test_rmse_runs", rmse)
        mlflow.log_metric("r2_real", test_r2_val)
        mlflow.log_metric("adjusted_r2", test_adjusted_r2)
        mlflow.log_metric("test_ccc", ccc)
        mlflow.log_metric("test_bias", bias)
        mlflow.log_metric("test_within_10", within_10)
        mlflow.log_metric("test_within_20", within_20)

        test_wicket_probs = tf.math.sigmoid(y_test_pred[1]).numpy().flatten()
        test_wide_probs = tf.math.sigmoid(y_test_pred[2]).numpy().flatten()

        wicket_rate = np.mean(y_train["wicket"])
        wide_rate = np.mean(y_train["wide"])

        wicket_thresh = float(np.percentile(test_wicket_probs, (1 - wicket_rate) * 100))
        wide_thresh = float(np.percentile(test_wide_probs, (1 - wide_rate) * 100))

        test_wicket_preds = (test_wicket_probs > wicket_thresh).astype(float)
        test_wide_preds = (test_wide_probs > wide_thresh).astype(float)

        mlflow.log_metric(
            "test_wicket_precision",
            precision_score(y_test["wicket"], test_wicket_preds, zero_division=0),
        )
        mlflow.log_metric(
            "test_wicket_recall",
            recall_score(y_test["wicket"], test_wicket_preds, zero_division=0),
        )
        mlflow.log_metric(
            "test_wicket_f1",
            f1_score(y_test["wicket"], test_wicket_preds, zero_division=0),
        )
        mlflow.log_metric(
            "test_wicket_mcc", matthews_corrcoef(y_test["wicket"], test_wicket_preds)
        )

        mlflow.log_metric(
            "test_wide_precision",
            precision_score(y_test["wide"], test_wide_preds, zero_division=0),
        )
        mlflow.log_metric(
            "test_wide_recall",
            recall_score(y_test["wide"], test_wide_preds, zero_division=0),
        )
        mlflow.log_metric(
            "test_wide_f1", f1_score(y_test["wide"], test_wide_preds, zero_division=0)
        )
        mlflow.log_metric(
            "test_wide_mcc", matthews_corrcoef(y_test["wide"], test_wide_preds)
        )

        sample_x, sample_y = train_gen[0]
        sample_pred = model.predict(sample_x, verbose=0)
        signature = infer_signature(sample_x, sample_pred)
        mlflow.tensorflow.log_model(
            model,
            name=f"{MODEL_TYPE}_model",
            signature=signature,
            pip_requirements=["tensorflow", "numpy", "pandas", "scikit-learn"],
        )

        mlflow.log_artifact(save_path)
        mlflow.log_artifact(args.config)

        logger.info("Training pipeline loop completed cleanly")

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

        test_mae = test_results["score_mae"]
        test_r2 = test_r2_val
        test_mse = mse
        test_rmse = rmse

        logger.info("MLOps Gate Assessment validation check started")
        logger.info(
            f"Current Metrics -> MAE: {test_mae} | MSE: {test_mse} | RMSE: {test_rmse} | R2: {test_r2}"
        )

        if (
            test_mae < GATE_MAE
            and test_r2 > GATE_R2
            and test_mse < GATE_MSE
            and test_rmse < GATE_RMSE
        ):
            logger.info(
                "[SUCCESS] Model strictly cleared target metrics gate verification."
            )
            logger.info(
                "Promoting architectural deployment package to Production registry..."
            )

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
            logger.warning(
                "[WARNING] Model parameters missed gate margins — deployment package preserved in Staging registry."
            )

    except Exception as e:
        mlflow.set_tag("status", "failed")
        logger.exception("MLOps script training thread failure encountered")
        raise
