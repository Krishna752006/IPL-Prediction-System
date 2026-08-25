import json
import os
import pickle
import time
import warnings
from datetime import datetime

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from core.config_loader import load_config
from core.logger import setup_logger
from core.model_bundle import IPLModelBundle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from training.loader import build_dataloaders
from training.tabtransformer_lstm import TabTransformerLSTM

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POSWEIGHT_CAPS = {
    "wicket_target": 10.0,
    "wide_target": 12.0,
}


def print_regression_metrics(name, y_true_norm, y_pred_norm, n_features=40):
    y_true = np.array(y_true_norm) * 180
    y_pred = np.array(y_pred_norm) * 180
    n = len(y_true)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def print_binary_metrics(y_true, y_pred, y_prob):
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Suppress MCC warnings for batches that might only have one class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc = float(matthews_corrcoef(y_true, y_pred))

    mse = float(mean_squared_error(y_true, y_prob))
    rmse = float(np.sqrt(mse))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "mse": mse,
        "rmse": rmse,
        "roc_auc": roc_auc,
    }


def compute_pos_weight(dataset, target_key):
    targets = [sample[target_key].item() for sample in dataset]
    pos = sum(targets)
    neg = len(targets) - pos
    raw = neg / (pos + 1e-6)
    cap = POSWEIGHT_CAPS[target_key]
    capped = min(raw, cap)
    return torch.tensor([capped])


def get_adaptive_threshold(probs, positive_rate):
    if len(probs) == 0:
        return 0.5
    return float(np.percentile(probs, (1 - positive_rate) * 100))


def get_batch_metrics(y_true, y_prob, thresh):
    """Helper to calculate per-batch metrics cleanly."""
    y_pred = (np.array(y_prob) > thresh).astype(float)

    # Using probabilities for MSE provides true Brier Score penalty
    mse = mean_squared_error(y_true, y_prob)
    rmse = np.sqrt(mse)

    f1 = f1_score(y_true, y_pred, zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc = matthews_corrcoef(y_true, y_pred)

    return mse, rmse, f1, mcc


def save_static_embeddings_to_json(
    model, player2idx, venue2idx, target_year, save_dir="./saved_seasons"
):
    os.makedirs(save_dir, exist_ok=True)
    embeddings_dict = {
        "players": {},
        "venues": {},
        "season": {},
        "match_state": model.match_state_embedding.weight.detach()
        .cpu()
        .numpy()
        .tolist(),
    }

    batter_weights = model.batter_embedding.weight.detach().cpu().numpy()
    non_striker_weights = model.non_striker_embedding.weight.detach().cpu().numpy()
    bowler_weights = model.bowler_embedding.weight.detach().cpu().numpy()

    for player, idx in player2idx.items():
        embeddings_dict["players"][player] = {
            "batter_embedding": batter_weights[idx].tolist(),
            "non_striker_embedding": non_striker_weights[idx].tolist(),
            "bowler_embedding": bowler_weights[idx].tolist(),
        }

    venue_weights = model.venue_embedding.weight.detach().cpu().numpy()
    for venue, idx in venue2idx.items():
        embeddings_dict["venues"][venue] = venue_weights[idx].tolist()

    season_weights = model.season_embedding.weight.detach().cpu().numpy()
    for s_idx in range(1, len(season_weights)):
        year = 2006 + s_idx
        embeddings_dict["season"][str(year)] = season_weights[s_idx].tolist()

    json_path = os.path.join(save_dir, f"static_embeddings_{target_year}.json")
    with open(json_path, "w") as f:
        json.dump(embeddings_dict, f)

    logger.info(f"Saved static JSON embeddings to {json_path}")
    return json_path


def evaluate_model(model, dataloader, wicket_thresh, wide_thresh, split_name="VAL"):
    if not dataloader or len(dataloader) == 0:
        return {}, {}, {}

    model.eval()
    all_score_preds, all_score_targets = [], []
    all_wicket_probs, all_wicket_targets = [], []
    all_wide_probs, all_wide_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            numerical = batch["numerical_features"].to(DEVICE)
            categorical = batch["categorical_features"].to(DEVICE)
            outputs = model(numerical, categorical)

            all_score_preds.extend(outputs["score"].cpu().numpy())
            all_score_targets.extend(batch["score_target"].cpu().numpy())
            all_wicket_probs.extend(torch.sigmoid(outputs["wicket"]).cpu().numpy())
            all_wicket_targets.extend(batch["wicket_target"].cpu().numpy())
            all_wide_probs.extend(torch.sigmoid(outputs["wide"]).cpu().numpy())
            all_wide_targets.extend(batch["wide_target"].cpu().numpy())

    all_wicket_preds = (
        (np.array(all_wicket_probs) > wicket_thresh).astype(float).tolist()
    )
    all_wide_preds = (np.array(all_wide_probs) > wide_thresh).astype(float).tolist()

    reg_metrics = print_regression_metrics(
        split_name, all_score_targets, all_score_preds
    )
    wicket_metrics = print_binary_metrics(
        all_wicket_targets, all_wicket_preds, all_wicket_probs
    )
    wide_metrics = print_binary_metrics(
        all_wide_targets, all_wide_preds, all_wide_probs
    )

    return reg_metrics, wicket_metrics, wide_metrics


def save_model_to_staging(model, config, target_year, dataset_version, feature_version):
    staging_dir = config["paths"]["staging_dir"]
    os.makedirs(staging_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"tabtransformer_lstm_season_{target_year}_{timestamp}.pkl"
    staging_path = os.path.join(staging_dir, model_filename)

    bundle = IPLModelBundle(
        model=model,
        dataset_version=dataset_version,
        feature_version=feature_version,
    )

    with open(staging_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Saved model bundle to staging: {staging_path}")
    return staging_path, model_filename


def train_one_year(
    config_path="configs/tabtransformer.yaml",
    manual_wicket_thresh=None,
    manual_wide_thresh=None,
):
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["paths"].get("mlflow_db", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config["experiment"].get("name", "ipl_tabtransformer_lstm"))

    target_year = config["data"]["target_year"]
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    lr = float(config["model"]["learning_rate"])
    weight_decay = float(config["model"]["weight_decay"])

    train_loader, val_loader, test_loader, train_dataset, _, _, _ = build_dataloaders(
        parquet_path=config["data"]["path"],
        players_json_path=config["data"].get("players_json", "data/all_players.json"),
        venues_json_path=config["data"].get("venues_json", "data/all_venues.json"),
        batch_size=batch_size,
        sequence_length=config["data"]["sequence_length"],
        target_year=target_year,
    )

    model = TabTransformerLSTM(
        num_players=train_dataset.num_players,
        num_venues=train_dataset.num_venues,
        num_seasons=train_dataset.num_seasons,
        numerical_dim=train_dataset.numerical_dim,
    ).to(DEVICE)

    wicket_pos_weight = compute_pos_weight(train_dataset, "wicket_target").to(DEVICE)
    wide_pos_weight = compute_pos_weight(train_dataset, "wide_target").to(DEVICE)

    run_criterion = nn.HuberLoss(reduction="none", delta=0.14)
    wicket_criterion = nn.BCEWithLogitsLoss(
        pos_weight=wicket_pos_weight, reduction="none"
    )
    wide_criterion = nn.BCEWithLogitsLoss(pos_weight=wide_pos_weight, reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    with mlflow.start_run(run_name=f"tabtransformer_lstm_{target_year}") as run:
        mlflow.log_params(config["model"])
        mlflow.log_params(config["data"])

        best_composite = 0.0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            all_score_preds, all_score_targets = [], []
            all_wicket_probs, all_wicket_targets = [], []
            all_wide_probs, all_wide_targets = [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                numerical = batch["numerical_features"].to(DEVICE)
                categorical = batch["categorical_features"].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(numerical, categorical)

                batch_seasons = categorical[:, -1, 4].float()
                max_train_season_id = float(target_year - 2007)
                recency_weights = 1.0 + 2.0 * (batch_seasons / max_train_season_id)

                run_loss = (
                    run_criterion(outputs["score"], batch["score_target"].to(DEVICE))
                    * recency_weights
                ).mean()
                wicket_loss = (
                    wicket_criterion(
                        outputs["wicket"], batch["wicket_target"].to(DEVICE)
                    )
                    * recency_weights
                ).mean()
                wide_loss = (
                    wide_criterion(outputs["wide"], batch["wide_target"].to(DEVICE))
                    * recency_weights
                ).mean()

                loss = 2 * run_loss + 1 * wicket_loss + 0.8 * wide_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                all_score_preds.extend(outputs["score"].detach().cpu().numpy())
                all_score_targets.extend(batch["score_target"].cpu().numpy())

                # Fetch detached arrays for per-batch metric calculation
                batch_w_probs = torch.sigmoid(outputs["wicket"]).detach().cpu().numpy()
                batch_w_targets = batch["wicket_target"].cpu().numpy()
                all_wicket_probs.extend(batch_w_probs)
                all_wicket_targets.extend(batch_w_targets)

                batch_wd_probs = torch.sigmoid(outputs["wide"]).detach().cpu().numpy()
                batch_wd_targets = batch["wide_target"].cpu().numpy()
                all_wide_probs.extend(batch_wd_probs)
                all_wide_targets.extend(batch_wd_targets)

                # Use manual thresholds if provided, otherwise default to 0.5 for active batch calculation
                current_w_thresh = (
                    manual_wicket_thresh if manual_wicket_thresh is not None else 0.5
                )
                current_wd_thresh = (
                    manual_wide_thresh if manual_wide_thresh is not None else 0.5
                )

                # Compute batch-level metrics
                _, _, w_f1, w_mcc = get_batch_metrics(
                    batch_w_targets, batch_w_probs, current_w_thresh
                )
                _, _, wd_f1, wd_mcc = get_batch_metrics(
                    batch_wd_targets, batch_wd_probs, current_wd_thresh
                )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "W_F1": f"{w_f1:.3f}",
                        "W_MCC": f"{w_mcc:.3f}",
                        "Wd_F1": f"{wd_f1:.3f}",
                        "Wd_MCC": f"{wd_mcc:.3f}",
                    }
                )

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Determine final epoch thresholds
            if manual_wicket_thresh is not None:
                wicket_thresh = manual_wicket_thresh
            else:
                wicket_rate = sum(all_wicket_targets) / max(len(all_wicket_targets), 1)
                wicket_thresh = get_adaptive_threshold(all_wicket_probs, wicket_rate)

            if manual_wide_thresh is not None:
                wide_thresh = manual_wide_thresh
            else:
                wide_rate = sum(all_wide_targets) / max(len(all_wide_targets), 1)
                wide_thresh = get_adaptive_threshold(all_wide_probs, wide_rate)

            reg_metrics, wicket_metrics, wide_metrics = evaluate_model(
                model, val_loader, wicket_thresh, wide_thresh, split_name="VAL"
            )

            mlflow.log_metric("val_r2", reg_metrics.get("r2", 0.0), step=epoch)
            mlflow.log_metric("val_mae", reg_metrics.get("mae", 0.0), step=epoch)
            mlflow.log_metric(
                "val_wicket_auc", wicket_metrics.get("roc_auc", 0.0), step=epoch
            )
            mlflow.log_metric(
                "val_wide_auc", wide_metrics.get("roc_auc", 0.0), step=epoch
            )

            composite = (
                0.55 * max(0.0, reg_metrics.get("r2", 0.0))
                + 0.20 * wicket_metrics.get("roc_auc", 0.0)
                + 0.25 * wide_metrics.get("roc_auc", 0.0)
            )

            if composite > best_composite:
                best_composite = composite
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final Test Evaluation
        test_reg, test_wicket, test_wide = evaluate_model(
            model, test_loader, wicket_thresh, wide_thresh, split_name="TEST"
        )
        for k, v in test_reg.items():
            mlflow.log_metric(f"test_{k}", v)

        # Save Artifacts & Staging Model
        embeddings_json_path = save_static_embeddings_to_json(
            model,
            train_dataset.player2idx,
            train_dataset.venue2idx,
            target_year,
            save_dir=config["paths"]["embeddings_json"],
        )
        mlflow.log_artifact(embeddings_json_path)

        staging_path, model_filename = save_model_to_staging(
            model=model,
            config=config,
            target_year=target_year,
            dataset_version=config["data"].get("dataset_version", "V1"),
            feature_version=config["data"].get("feature_version", "tabtransformer_v1"),
        )
        mlflow.log_artifact(staging_path)

        logger.info(
            f"Run completed successfully. Model saved to staging as {model_filename}"
        )


if __name__ == "__main__":
    # You can now explicitly pass manual thresholds here to override adaptive logic.
    train_one_year(
        config_path="configs/tabtransformer.yaml",
        manual_wicket_thresh=None,
        manual_wide_thresh=None,
    )
