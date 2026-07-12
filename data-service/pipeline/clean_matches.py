import json
from pathlib import Path

import numpy as np
import pandas as pd
from config import CLEAN_MATCHES_PATH, RAW_MATCHES


def clean_matches():

    print("Loading matches dataset...")
    matches = pd.read_csv(RAW_MATCHES)

    print("Fixing season format...")
    matches["season"] = matches["season"].apply(
        lambda x: (
            int(x.split("/")[0])
            if x == "2020/21"
            else int("20" + x.split("/")[1]) if "/" in x else int(x)
        )
    )

    print("Handling missing values...")
    matches["winner_runs"] = matches["winner_runs"].fillna(0)
    matches["winner_wickets"] = matches["winner_wickets"].fillna(0)

    matches["winner"] = matches["winner"].fillna(matches["eliminator"])

    print("Removing D/L method matches...")
    matches = matches[matches["method"] != "D/L"]

    print("Dropping unnecessary columns...")
    matches.drop(
        [
            "event",
            "umpire2",
            "toss_winner",
            "neutralvenue",
            "umpire1",
            "reserve_umpire",
            "match_referee",
            "tv_umpire",
            "eliminator",
            "date1",
            "date2",
            "method",
            "gender",
            "balls_per_over",
            "outcome",
            "city",
            "match_number",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("Removing matches without winner...")
    matches = matches[matches["winner"].notna()]

    with open("../New Data/data/venue.json", "r") as file:
        venues = json.load(file)

    print("Mapping Venues...")
    matches["venue"] = matches["venue"].map(venues)

    print("Converting types...")
    matches["date"] = pd.to_datetime(matches["date"])
    matches["winner_runs"] = matches["winner_runs"].astype(int)
    matches["winner_wickets"] = matches["winner_wickets"].astype(int)

    print("Sorting by date...")
    matches = matches.sort_values("date")

    print("Adding Match State context...")
    matches["season_match_no"] = matches.groupby("season").cumcount() + 1
    matches["season_total_matches"] = matches.groupby("season")[
        "season_match_no"
    ].transform("max")

    matches_remaining = matches["season_total_matches"] - matches["season_match_no"]
    league_total = np.where(
        matches["season"] > 2010,
        matches["season_total_matches"] - 4,
        matches["season_total_matches"] - 3,
    )
    progress = matches["season_match_no"] / league_total

    conditions = [
        matches_remaining == 0,
        matches_remaining < 4,
        progress <= 0.50,
        progress <= 0.75,
    ]
    choices = ["Final", "Playoffs", "Starting", "Middle"]
    matches["match_state"] = np.select(conditions, choices, default="Business_End")
    matches.drop(columns=["season_match_no", "season_total_matches"], inplace=True)

    print("Checking if Folder is present...")
    CLEAN_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving parquet...")
    matches.to_parquet(CLEAN_MATCHES_PATH, index=False)

    print("Clean matches saved successfully.")


if __name__ == "__main__":
    clean_matches()
