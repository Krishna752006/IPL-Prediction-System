import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import IPLDataset
from loader import build_dataloaders
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tabtransformer_lstm import (
    TabTransformerLSTM,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open("data/batter_classification.json", "r") as f:
        BATTER_CLASSES = json.load(f)
    with open("data/bowler_classification.json", "r") as f:
        BOWLER_CLASSES = json.load(f)
except FileNotFoundError:
    print("Warning: Classification JSONs not found. Update paths if necessary.")
    BATTER_CLASSES, BOWLER_CLASSES = {}, {}

PLAYER_TO_BATTER_ROLE = {
    p: role for role, players in BATTER_CLASSES.items() for p in players
}
PLAYER_TO_BOWLER_ROLE = {
    p: role for role, players in BOWLER_CLASSES.items() for p in players
}


def print_regression_metrics(name, y_true_norm, y_pred_norm, n_features=40):

    y_true = np.array(y_true_norm) * 180
    y_pred = np.array(y_pred_norm) * 180

    n = len(y_true)

    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

    from scipy.stats import pearsonr, spearmanr

    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)

    bias = np.mean(y_pred - y_true)
    within_10 = np.mean(np.abs(y_pred - y_true) <= 10) * 100
    within_20 = np.mean(np.abs(y_pred - y_true) <= 20) * 100

    print(f"\n{name} SCORE METRICS")
    print(f"  MAE          : {mae:.2f} runs")
    print(f"  MSE          : {mse:.2f}")
    print(f"  RMSE         : {rmse:.2f} runs")
    print(f"  R²           : {r2:.4f}")
    print(f"  Adjusted R²  : {adj_r2:.4f}  (n_features={n_features})")
    print(f"  ── Correlation metrics ──────────────────────")
    print(f"  CCC          : {ccc:.4f}  ← regression MCC equivalent")
    print(f"  Pearson r    : {pearson:.4f}")
    print(f"  Spearman r   : {spearman:.4f}")
    print(f"  ── Calibration ──────────────────────────────")
    print(
        f"  Bias         : {bias:+.2f} runs  ({'over' if bias > 0 else 'under'}-predicting)"
    )
    print(f"  Within ±10   : {within_10:.1f}%")
    print(f"  Within ±20   : {within_20:.1f}%")

    return r2, ccc, mae, mse


def print_binary_metrics(
    name,
    y_true,
    y_pred,
    y_prob,
):

    print(f"\n{name} METRICS")

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

    except ValueError:
        roc_auc = 0.0

    print(
        "Accuracy:",
        round(accuracy, 4),
    )

    print(
        "Precision:",
        round(precision, 4),
    )

    print(
        "Recall:",
        round(recall, 4),
    )

    print(
        "F1:",
        round(f1, 4),
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4),
    )

    print(
        "MCC:",
        round(mcc, 4),
    )


POSWEIGHT_CAPS = {
    "wicket_target": 10.0,
    "wide_target": 12.0,
}


def compute_pos_weight(dataset, target_key):
    targets = [sample[target_key].item() for sample in dataset]
    pos = sum(targets)
    neg = len(targets) - pos
    raw = neg / (pos + 1e-6)
    cap = POSWEIGHT_CAPS[target_key]
    capped = min(raw, cap)
    print(f"  {target_key}: raw={raw:.1f}, cap={cap}, used={capped:.1f}")
    return torch.tensor([capped])


def get_adaptive_threshold(probs, positive_rate):
    if len(probs) == 0:
        return 0.5
    return float(np.percentile(probs, (1 - positive_rate) * 100))


def inject_debutant_embeddings(model, train_dataset, noise_std=0.05):
    print("\n" + "=" * 40)
    print("🏏 INJECTING DEBUTANT EMBEDDINGS")
    print("=" * 40)

    idx2player = {v: k for k, v in train_dataset.player2idx.items()}
    idx2venue = {v: k for k, v in train_dataset.venue2idx.items()}

    seen_batters = set(np.unique(train_dataset.X_categorical[:, :, 0]))
    seen_non_strikers = set(np.unique(train_dataset.X_categorical[:, :, 1]))
    seen_bowlers = set(np.unique(train_dataset.X_categorical[:, :, 2]))
    seen_venues = set(np.unique(train_dataset.X_categorical[:, :, 3]))
    seen_states = set(np.unique(train_dataset.X_categorical[:, :, 5]))

    configs = [
        (
            "Batter",
            model.batter_embedding,
            seen_batters,
            set(train_dataset.player2idx.values()),
            idx2player,
            PLAYER_TO_BATTER_ROLE,
        ),
        (
            "Non-Striker",
            model.non_striker_embedding,
            seen_non_strikers,
            set(train_dataset.player2idx.values()),
            idx2player,
            PLAYER_TO_BATTER_ROLE,
        ),
        (
            "Bowler",
            model.bowler_embedding,
            seen_bowlers,
            set(train_dataset.player2idx.values()),
            idx2player,
            PLAYER_TO_BOWLER_ROLE,
        ),
        (
            "Venue",
            model.venue_embedding,
            seen_venues,
            set(train_dataset.venue2idx.values()),
            idx2venue,
            None,
        ),
        (
            "MatchState",
            model.match_state_embedding,
            seen_states,
            set(range(model.match_state_embedding.num_embeddings)),
            None,
            None,
        ),
    ]

    with torch.no_grad():
        for name, embed_layer, seen_set, all_ids, idx2name, role_map in configs:
            weights = embed_layer.weight.data
            unseen_ids = all_ids - seen_set

            # Use role-based averages if a role map is provided; otherwise, use a global average
            if role_map is not None:
                role_sums, role_counts = {}, {}
                for seen_id in seen_set:
                    if seen_id == 0:
                        continue

                    # Guard against missing keys in idx2name
                    entity_name = idx2name.get(seen_id)
                    role = role_map.get(entity_name, "Unknown")

                    if role not in role_sums:
                        role_sums[role] = torch.zeros_like(weights[0])
                        role_counts[role] = 0

                    role_sums[role] += weights[seen_id]
                    role_counts[role] += 1

                role_averages = {r: role_sums[r] / role_counts[r] for r in role_sums}
                global_avg = (
                    weights[list(seen_set)].mean(dim=0)
                    if seen_set
                    else torch.zeros_like(weights[0])
                )
            else:
                global_avg = (
                    weights[list(seen_set)].mean(dim=0)
                    if seen_set
                    else torch.zeros_like(weights[0])
                )

            injected_count = 0
            for unseen_id in unseen_ids:
                if unseen_id == 0:
                    continue

                if role_map is not None:
                    entity_name = idx2name.get(unseen_id)
                    role = role_map.get(entity_name, "Unknown")
                    base_embed = role_averages.get(role, global_avg)
                else:
                    base_embed = global_avg

                noise = torch.randn_like(base_embed) * noise_std
                weights[unseen_id] = base_embed.clone() + noise
                injected_count += 1

            print(
                f"  [{name}] Processed {len(seen_set)} seen | Injected {injected_count} unseen"
            )


def save_static_embeddings_to_json(
    model, player2idx, venue2idx, target_year, save_dir="./saved_seasons"
):
    """Extracts raw static embeddings from the model and saves them as a JSON dictionary."""
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

    # Extract Player Embeddings
    batter_weights = model.batter_embedding.weight.detach().cpu().numpy()
    non_striker_weights = model.non_striker_embedding.weight.detach().cpu().numpy()
    bowler_weights = model.bowler_embedding.weight.detach().cpu().numpy()

    for player, idx in player2idx.items():
        embeddings_dict["players"][player] = {
            "batter_embedding": batter_weights[idx].tolist(),
            "non_striker_embedding": non_striker_weights[idx].tolist(),
            "bowler_embedding": bowler_weights[idx].tolist(),
        }

    # Extract Venue Embeddings
    venue_weights = model.venue_embedding.weight.detach().cpu().numpy()
    for venue, idx in venue2idx.items():
        embeddings_dict["venues"][venue] = venue_weights[idx].tolist()

    # Extract Season Embeddings (Assuming 1-indexed from 2007)
    season_weights = model.season_embedding.weight.detach().cpu().numpy()
    for s_idx in range(1, len(season_weights)):
        year = 2006 + s_idx
        embeddings_dict["season"][str(year)] = season_weights[s_idx].tolist()

    json_path = os.path.join(save_dir, f"static_embeddings_{target_year}.json")
    with open(json_path, "w") as f:
        json.dump(embeddings_dict, f)

    print(f"\nSaved static JSON embeddings to {json_path}")


def inject_season_embedding(model, train_dataset, target_year, noise_std):
    target_season_id = target_year - 2007 + 1

    seen_season_ids = set(np.unique(train_dataset.X_categorical[:, :, 4]))
    seen_season_ids.discard(0)
    seen_season_ids.discard(target_season_id)

    weights = model.season_embedding.weight.data

    avg_embedding = torch.stack([weights[sid] for sid in seen_season_ids]).mean(dim=0)

    with torch.no_grad():
        noise = torch.randn_like(avg_embedding) * noise_std
        weights[target_season_id] = avg_embedding + noise

    print(
        f"  Season {target_year} (id={target_season_id}): "
        f"injected from avg of {len(seen_season_ids)} training seasons"
    )


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def evaluate_model(
    model, dataloader, wicket_thresh, wide_thresh, split_name="VAL", return_score=False
):
    if not dataloader or len(dataloader) == 0:
        print(f"\nNo data for {split_name} split.")
        return

    print(f"\n--- {split_name} EVALUATION ---")
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

    naive_mae = np.mean(np.abs((0.825 * 180) - (np.array(all_score_targets) * 180)))
    print(f"Naive constant mean MAE: {naive_mae:.2f} runs")

    eval_r2, _, _, _ = print_regression_metrics(
        split_name, all_score_targets, all_score_preds
    )

    print_binary_metrics(
        "WICKET", all_wicket_targets, all_wicket_preds, all_wicket_probs
    )
    print_binary_metrics("WIDE", all_wide_targets, all_wide_preds, all_wide_probs)

    if split_name == "TEST":
        print("\nRunning MC Dropout for Uncertainty Intervals...")
        enable_dropout(model)
        mc_preds = []
        for _ in range(10):
            batch_preds = []
            with torch.no_grad():
                for batch in dataloader:
                    numerical = batch["numerical_features"].to(DEVICE)
                    categorical = batch["categorical_features"].to(DEVICE)
                    outputs = model(numerical, categorical)
                    batch_preds.extend(outputs["score"].cpu().numpy())
            mc_preds.append(batch_preds)

        mc_preds = np.array(mc_preds) * 180
        std_dev = np.std(mc_preds, axis=0)
        print(f"Average prediction uncertainty (std dev): ±{np.mean(std_dev):.2f} runs")

        model.eval()

    if return_score:
        try:
            wicket_auc = roc_auc_score(all_wicket_targets, all_wicket_probs)
        except ValueError:
            wicket_auc = 0.0

        try:
            wide_auc = roc_auc_score(all_wide_targets, all_wide_probs)
        except ValueError:
            wide_auc = 0.0

        return 0.55 * max(0.0, eval_r2) + 0.20 * wicket_auc + 0.25 * wide_auc

    return all_score_preds, all_score_targets


def train_one_year(
    target_year,
    epochs=10,
):

    print(f"\nTraining for season " f"{target_year}")

    train_loader, val_loader, test_loader, train_dataset, _, _, raw_df = (
        build_dataloaders(
            parquet_path=("../ml-service/data/" "processed/ab/" "features.parquet"),
            batch_size=200 * ((target_year - 2007) // 3),
            sequence_length=30,
            target_year=target_year,
        )
    )

    model = TabTransformerLSTM(
        num_players=train_dataset.num_players,
        num_venues=train_dataset.num_venues,
        num_seasons=train_dataset.num_seasons,
        numerical_dim=train_dataset.numerical_dim,
    ).to(DEVICE)

    print("\nCalculating class weights for imbalanced tasks...")
    wicket_pos_weight = compute_pos_weight(train_dataset, "wicket_target").to(DEVICE)
    wide_pos_weight = compute_pos_weight(train_dataset, "wide_target").to(DEVICE)

    run_criterion = nn.HuberLoss(reduction="none", delta=0.14)

    wicket_criterion = nn.BCEWithLogitsLoss(
        pos_weight=wicket_pos_weight, reduction="none"
    )

    wide_criterion = nn.BCEWithLogitsLoss(pos_weight=wide_pos_weight, reduction="none")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    best_composite_score = 0.0
    best_model_state = None
    early_stop_patience = 5
    no_improve_count = 0

    max_train_season_id = float(target_year - 2007)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
        )

        all_score_preds = []
        all_score_targets = []

        all_wicket_probs = []
        all_wicket_targets = []

        all_wide_probs = []
        all_wide_targets = []

        for batch in loop:
            numerical = batch["numerical_features"].to(DEVICE)
            categorical = batch["categorical_features"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(numerical, categorical)

            batch_seasons = categorical[:, -1, 4].float()

            max_train_season_id = float(target_year - 2007)
            recency_weights = 1.0 + 2.0 * (batch_seasons / max_train_season_id)

            run_loss_raw = run_criterion(
                outputs["score"], batch["score_target"].to(DEVICE)
            )
            wicket_loss_raw = wicket_criterion(
                outputs["wicket"], batch["wicket_target"].to(DEVICE)
            )
            wide_loss_raw = wide_criterion(
                outputs["wide"], batch["wide_target"].to(DEVICE)
            )

            run_loss = (run_loss_raw * recency_weights).mean()
            wicket_loss = (wicket_loss_raw * recency_weights).mean()
            wide_loss = (wide_loss_raw * recency_weights).mean()

            loss = 2 * run_loss + 1 * wicket_loss + 0.8 * wide_loss

            all_score_preds.extend(outputs["score"].detach().cpu().numpy())
            all_score_targets.extend(batch["score_target"].cpu().numpy())

            all_wicket_probs.extend(
                torch.sigmoid(outputs["wicket"]).detach().cpu().numpy()
            )
            all_wicket_targets.extend(batch["wicket_target"].cpu().numpy())

            all_wide_probs.extend(torch.sigmoid(outputs["wide"]).detach().cpu().numpy())
            all_wide_targets.extend(batch["wide_target"].cpu().numpy())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=round(
                    loss.item(),
                    4,
                )
            )

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch " f"{epoch+1}" f" Average Loss:" f" {avg_loss:.4f}")

        wicket_rate = sum(all_wicket_targets) / max(len(all_wicket_targets), 1)
        wide_rate = sum(all_wide_targets) / max(len(all_wide_targets), 1)

        print(f"Wicket rate: {wicket_rate:.3f}")

        wicket_thresh = get_adaptive_threshold(all_wicket_probs, wicket_rate)
        wide_thresh = get_adaptive_threshold(all_wide_probs, wide_rate)

        print(
            f"Adaptive thresholds → wicket:{wicket_thresh:.3f}  wide:{wide_thresh:.3f}"
        )

        all_wicket_preds = (
            (np.array(all_wicket_probs) > wicket_thresh).astype(float).tolist()
        )
        all_wide_preds = (np.array(all_wide_probs) > wide_thresh).astype(float).tolist()

        if (epoch + 1) % 2 == 0:

            r2, ccc, mae, mse = print_regression_metrics(
                "TRAIN", all_score_targets, all_score_preds
            )
            score_component = max(0.0, r2)

            try:
                wicket_auc = roc_auc_score(all_wicket_targets, all_wicket_probs)
            except ValueError:
                wicket_auc = 0.0

            try:
                wide_auc = roc_auc_score(all_wide_targets, all_wide_probs)
            except ValueError:
                wide_auc = 0.0

            current_composite = (
                0.55 * score_component + 0.20 * wicket_auc + 0.25 * wide_auc
            )

            print_binary_metrics(
                "TRAIN WICKET", all_wicket_targets, all_wicket_preds, all_wicket_probs
            )
            print_binary_metrics(
                "TRAIN WIDE", all_wide_targets, all_wide_preds, all_wide_probs
            )

            if current_composite > best_composite_score:
                best_composite_score = current_composite
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f" ★ New best TRAIN composite: {current_composite:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  No improvement ({no_improve_count}/{early_stop_patience})")
                if no_improve_count >= early_stop_patience:
                    print(
                        f"  ⚠ Early stopping triggered at epoch {epoch+1} — training plateaued"
                    )
                    break

            print("Wicket prob mean:", np.mean(all_wicket_probs))
            print("Wide prob mean:", np.mean(all_wide_probs))

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (composite score: {best_composite_score:.4f})")

    inject_debutant_embeddings(model, train_dataset, noise_std=0.05)
    inject_season_embedding(model, train_dataset, target_year, noise_std=0.05)

    val_preds, val_targets = evaluate_model(
        model, val_loader, wicket_thresh, wide_thresh, split_name="VALIDATION"
    )
    test_preds, test_targets = evaluate_model(
        model, test_loader, wicket_thresh, wide_thresh, split_name="TEST"
    )

    save_static_embeddings_to_json(
        model=model,
        player2idx=train_dataset.player2idx,
        venue2idx=train_dataset.venue2idx,
        target_year=target_year,
    )


if __name__ == "__main__":

    for season in range(
        2010,
        2011,
    ):

        print("\n" + "=" * 50)

        print(f"Starting training " f"for season {season}")

        start_time = time.time()

        train_one_year(
            target_year=season,
            epochs=1,
        )

        end_time = time.time()

        total_seconds = end_time - start_time

        minutes = int(total_seconds // 60)

        seconds = total_seconds % 60

        print(
            f"\nSeason {season} "
            f"completed in "
            f"{minutes} min "
            f"{seconds:.2f} sec"
        )

        print("=" * 50)
