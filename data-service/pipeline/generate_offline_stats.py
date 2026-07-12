import json
import os

import pandas as pd
from config import (
    BATTER_FORM_PATH,
    BOWLER_FORM_PATH,
    CLEAN_DELIVERIES_PATH,
    CLEAN_MATCHES_PATH,
    OFFLINE_STATS_DIR,
    VENUE_PHASE_AVG_PATH,
)

os.makedirs(OFFLINE_STATS_DIR, exist_ok=True)


def generate_offline_stats():
    print("Loading clean datasets...")
    balls = pd.read_parquet(CLEAN_DELIVERIES_PATH)
    matches = pd.read_parquet(CLEAN_MATCHES_PATH)

    df = balls.merge(matches[["matchId", "venue", "season"]], on="matchId", how="left")

    print("Bucketing the Seasons...")
    df["season_bucket"] = pd.cut(
        df["season"],
        bins=[2007, 2013, 2019, 2022, 2025],
        labels=[
            "2008_2013",
            "2014_2019",
            "2020_2022",
            "2023_2025",
        ],
    )

    print("Calculating Venue Phase Averages...")

    df["phase"] = pd.cut(
        df["over"],
        bins=[-1, 5.9, 14.9, 20],
        labels=["PP", "Middle", "Death"],
    )

    venue_phase_stats = df.groupby(
        ["season_bucket", "venue", "phase"],
        observed=True,
    ).agg(
        runs=("total_runs", "sum"),
        balls=("total_runs", "count"),
    )

    venue_phase_stats["runs_per_over"] = (
        venue_phase_stats["runs"] / venue_phase_stats["balls"]
    ) * 6

    league_phase_rpo = df.groupby(
        ["season_bucket", "phase"],
        observed=True,
    ).agg(
        runs=("total_runs", "sum"),
        balls=("total_runs", "count"),
    )

    league_phase_rpo["league_rpo"] = (
        league_phase_rpo["runs"] / league_phase_rpo["balls"]
    ) * 6

    venue_phase_stats = venue_phase_stats.join(
        league_phase_rpo["league_rpo"], on=["season_bucket", "phase"]
    )

    venue_phase_stats["scoring_index"] = (
        venue_phase_stats["runs_per_over"] / venue_phase_stats["league_rpo"]
    )

    venue_phase_stats.to_json(VENUE_PHASE_AVG_PATH, orient="index")

    print("Calculating Recent Form...")

    batter_match = (
        df.groupby(["batsman", "date", "matchId"])
        .agg(runs=("batsman_runs", "sum"), balls=("matchId", "count"))
        .reset_index()
        .sort_values(["batsman", "date", "matchId"])
    )

    for i in range(1, 4):
        batter_match[f"last_{i}_runs"] = (
            batter_match.groupby("batsman")["runs"].shift(i).fillna(0)
        )

        batter_match[f"last_{i}_balls"] = (
            batter_match.groupby("batsman")["balls"].shift(i).fillna(0)
        )

    bowler_match = (
        df.groupby(["bowler", "date", "matchId"])
        .agg(runs_conceded=("total_runs", "sum"), balls_bowled=("matchId", "count"))
        .reset_index()
        .sort_values(["bowler", "date", "matchId"])
    )

    for i in range(1, 4):
        bowler_match[f"last_{i}_runs_conceded"] = (
            bowler_match.groupby("bowler")["runs_conceded"].shift(i).fillna(0)
        )

        bowler_match[f"last_{i}_balls_bowled"] = (
            bowler_match.groupby("bowler")["balls_bowled"].shift(i).fillna(0)
        )

    batter_match["history_matches"] = batter_match.groupby("batsman").cumcount()

    bowler_match["history_matches"] = bowler_match.groupby("bowler").cumcount()

    batter_cols = [
        "matchId",
        "batsman",
        "history_matches",
    ]

    for i in range(1, 4):
        batter_cols.extend(
            [
                f"last_{i}_runs",
                f"last_{i}_balls",
            ]
        )

    batter_match[batter_cols].to_parquet(
        BATTER_FORM_PATH,
        index=False,
    )

    bowler_cols = [
        "matchId",
        "bowler",
        "history_matches",
    ]

    for i in range(1, 4):
        bowler_cols.extend(
            [
                f"last_{i}_runs_conceded",
                f"last_{i}_balls_bowled",
            ]
        )

    bowler_match[bowler_cols].to_parquet(
        BOWLER_FORM_PATH,
        index=False,
    )

    print("Offline precomputations complete!")


if __name__ == "__main__":
    generate_offline_stats()
