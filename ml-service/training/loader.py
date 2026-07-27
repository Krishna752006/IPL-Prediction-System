import json
import pandas as pd
import torch
from training.dataset import IPLDataset
from torch.utils.data import DataLoader


def build_dataloaders(
    parquet_path: str,
    players_json_path: str = "data/all_players.json",
    venues_json_path: str = "data/all_venues.json",
    batch_size: int = 64,
    sequence_length: int = 30,
    target_year: int = 2025,
    num_workers: int = 0,
):
    print("Loading parquet dataset...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataframe shape: {df.shape}")

    with open(players_json_path, "r") as f:
        all_players = json.load(f)
        player_map = {player: idx + 1 for idx, player in enumerate(all_players)}

    with open(venues_json_path, "r") as f:
        all_venues = json.load(f)
        venue_map = {venue: idx + 1 for idx, venue in enumerate(all_venues)}

    print("\nBuilding TRAIN dataset...")
    train_dataset = IPLDataset(
        df=df,
        target_year=target_year,
        player_mapping=player_map,
        venue_mapping=venue_map,
        mode="train",
        sequence_length=sequence_length,
    )

    print("\nBuilding VALIDATION dataset...")
    val_dataset = IPLDataset(
        df=df,
        target_year=target_year,
        player_mapping=player_map,
        venue_mapping=venue_map,
        mode="val",
        sequence_length=sequence_length,
    )

    print("\nBuilding TEST dataset...")
    test_dataset = IPLDataset(
        df=df,
        target_year=target_year,
        player_mapping=player_map,
        venue_mapping=venue_map,
        mode="test",
        sequence_length=sequence_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("\n==============================")
    print("DATALOADER SUMMARY")
    print("==============================")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {sequence_length}")
    print("==============================\n")

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
        df,
    )