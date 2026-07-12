import ast
import json

import numpy as np
import pandas as pd
from config import CLEAN_DELIVERIES_PATH, CLEAN_MATCHES_PATH, VERSION_DIR

FEATURES_PATH = VERSION_DIR / "features.parquet"

from config import (
    BATTER_FORM_PATH,
    BOWLER_FORM_PATH,
    VENUE_PHASE_AVG_PATH,
)


def build_features():

    print("Loading clean datasets...")
    balls = pd.read_parquet(CLEAN_DELIVERIES_PATH)
    matches = pd.read_parquet(CLEAN_MATCHES_PATH)

    print("Initial shapes:", balls.shape, matches.shape)

    balls = balls.sort_values(["matchId", "inning", "over", "total_balls"]).reset_index(
        drop=True
    )

    print("Fixing no-ball anomalies...")
    mask = balls["isNoBall"] > 1
    balls.loc[mask, "batsman_runs"] += balls.loc[mask, "isNoBall"] - 1
    balls.loc[mask, "isNoBall"] = 1

    print("Expanding wides...")
    balls["repeat"] = np.where(balls["isWide"] > 0, balls["isWide"], 1)
    balls = balls.loc[balls.index.repeat(balls["repeat"])].copy()
    balls.loc[balls["isWide"] > 0, "isWide"] = 1
    balls.drop(columns=["repeat"], inplace=True)

    print("Recomputing legal balls...")
    balls["is_legal"] = ((balls["isWide"] == 0) & (balls["isNoBall"] == 0)).astype(int)

    running_legal_count = balls.groupby(["matchId", "inning", "over"])[
        "is_legal"
    ].cumsum()

    balls["legal_ball"] = (
        running_legal_count.groupby([balls["matchId"], balls["inning"], balls["over"]])
        .shift(1)
        .fillna(0)
        + 1
    )

    balls = balls[balls["legal_ball"] <= 6].reset_index(drop=True)

    print("Basic match features...")
    balls["legal_ball_1"] = (balls["isWide"] == 0) & (balls["isNoBall"] == 0)

    balls["over_number"] = balls["over"].astype(int) + 1

    balls["phase_pp"] = (balls["over_number"] <= 6).astype(int)
    balls["phase_middle"] = (
        (balls["over_number"] > 6) & (balls["over_number"] <= 15)
    ).astype(int)
    balls["phase_death"] = (balls["over_number"] > 15).astype(int)

    print("Score + wickets...")
    balls["total_runs"] = (
        balls["batsman_runs"]
        + balls["isWide"]
        + balls["isNoBall"]
        + balls["Byes"]
        + balls["LegByes"]
        + balls["Penalty"]
    )
    balls["current_score"] = balls.groupby(["matchId", "inning"])["total_runs"].cumsum()

    balls.loc[balls["Penalty"] == 5, "batsman_runs"] = 5
    balls["batsman_runs"] = balls["batsman_runs"] + balls["Byes"] + balls["LegByes"]

    balls = balls.reset_index(drop=True)
    balls["is_wicket"] = (balls["player_dismissed"] != "Not Out").astype(int)
    balls["wickets_fallen"] = balls.groupby(["matchId", "inning"])["is_wicket"].cumsum()

    print("Target creation...")
    first_innings_score = (
        balls[balls["inning"] == 0].groupby("matchId")["current_score"].max()
    )

    balls["target"] = balls["matchId"].map(first_innings_score)
    balls.loc[balls["inning"] == 1, "target"] += 1
    balls.loc[balls["inning"] == 0, "target"] = 0

    balls["total_balls"] = balls.groupby(["matchId", "inning", "over"]).cumcount() + 1

    print("Applying manual fixes...")
    balls.loc[
        (balls["matchId"] == 1254073)
        & (balls["inning"] == 1)
        & (balls["over"] == 16)
        & (balls["total_balls"] == 5),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [3, 4, 181]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 1254073)
            & (balls["inning"] == 1)
            & (balls["over"] == 16)
            & (balls["total_balls"] > 5)
        ].index
    )

    balls.loc[
        (balls["matchId"] == 1178398)
        & (balls["inning"] == 1)
        & (balls["over"] == 17)
        & (balls["total_balls"] == 5),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [2, 3, 111]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 1178398)
            & (balls["inning"] == 1)
            & (balls["over"] == 17)
            & (balls["total_balls"] > 5)
        ].index
    )

    balls.loc[
        (balls["matchId"] == 729309)
        & (balls["inning"] == 1)
        & (balls["over"] == 18)
        & (balls["total_balls"] == 4),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [6, 6, 131]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 729309)
            & (balls["inning"] == 1)
            & (balls["over"] == 18)
            & (balls["total_balls"] > 4)
        ].index
    )

    balls = balls.sort_values(["matchId", "inning", "over", "total_balls"]).reset_index(
        drop=True
    )

    print("NoBall adjustments...")
    mask = (balls["isNoBall"] == 1) & (balls["player_dismissed"] != "Not Out")
    balls.loc[mask, "isWide"] = 1
    balls.loc[mask, "isNoBall"] = 0

    print("Target Creations...")
    balls["isWide_target"] = balls["isWide"].astype(int)
    balls["is_wicket_target"] = balls["is_wicket"].astype(int)

    balls["score_before"] = (
        balls.groupby(["matchId", "inning"])["current_score"].shift(1).fillna(0)
    )
    balls["wickets_before"] = (
        balls.groupby(["matchId", "inning"])["wickets_fallen"].shift(1).fillna(0)
    )

    print("Target progress...")
    balls["percentage_target_achieved"] = np.where(
        balls["inning"] == 0, 0.0, balls["score_before"] / balls["target"]
    )

    balls["percentage_target_achieved"] = (
        balls["percentage_target_achieved"].replace([np.inf, -np.inf], 0).fillna(0)
    )

    print("Merging match metadata and toss features...")
    balls = balls.merge(
        matches[["matchId", "venue", "match_state", "toss_decision"]],
        on="matchId",
        how="left",
    )

    balls["batting_first"] = (balls["inning"] == 0).astype(int)
    balls["toss_won"] = 0
    balls.loc[
        (balls["batting_first"] == 1) & (balls["toss_decision"] == "bat"), "toss_won"
    ] = 1
    balls.loc[
        (balls["batting_first"] == 0) & (balls["toss_decision"] == "field"), "toss_won"
    ] = 1

    print("Encoding match state features...")
    state_map = {
        "Starting": 1,
        "Middle": 2,
        "Business_End": 3,
        "Playoffs": 4,
        "Final": 5,
    }
    balls["match_state_id"] = balls["match_state"].map(state_map).fillna(1).astype(int)

    balls.drop(columns=["match_state", "toss_decision", "batting_first"], inplace=True)

    print("Merging Offline Stats (Form, H2H, Venue)...")

    season_map = matches.set_index("matchId")["season"]
    balls["season"] = balls["matchId"].map(season_map)

    batter_form = pd.read_parquet(BATTER_FORM_PATH)
    bowler_form = pd.read_parquet(BOWLER_FORM_PATH)

    batter_form = batter_form.rename(
        columns={"history_matches": "batter_history_matches"}
    )
    bowler_form = bowler_form.rename(
        columns={"history_matches": "bowler_history_matches"}
    )

    balls = balls.merge(
        batter_form,
        on=["matchId", "batsman"],
        how="left",
    )

    balls = balls.merge(
        bowler_form,
        on=["matchId", "bowler"],
        how="left",
    )

    form_cols = [
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
    balls[form_cols] = balls[form_cols].fillna(0)

    with open(VENUE_PHASE_AVG_PATH, "r") as f:
        venue_avg_raw = json.load(f)

    venue_phase_dict = {
        ast.literal_eval(key): float(stats["scoring_index"])
        for key, stats in venue_avg_raw.items()
    }

    balls["temp_phase"] = pd.cut(
        balls["over"], bins=[-1, 5.9, 14.9, 20], labels=["PP", "Middle", "Death"]
    ).astype(str)

    balls["season_bucket"] = pd.cut(
        balls["season"],
        bins=[2007, 2013, 2019, 2022, 2025],
        labels=["2008_2013", "2014_2019", "2020_2022", "2023_2025"],
    ).astype(str)

    balls["venue_phase_avg"] = (
        balls.set_index(["season_bucket", "venue", "temp_phase"])
        .index.map(venue_phase_dict)
        .fillna(1.0)
    )
    balls.drop(columns=["temp_phase", "season_bucket"], inplace=True)

    print("Run rate features...")
    TOTAL_BALLS = 120

    balls["balls_bowled"] = balls.groupby(["matchId", "inning"])["legal_ball_1"].cumsum() - balls["legal_ball_1"]
    
    balls["balls_remaining"] = TOTAL_BALLS - balls["balls_bowled"]

    balls["overs_bowled"] = balls["balls_bowled"] / 6

    balls["current_run_rate"] = np.where(
        balls["balls_bowled"] > 0, balls["score_before"] / balls["overs_bowled"], 0
    )

    balls["runs_required"] = balls["target"] - balls["score_before"]
    balls["required_run_rate"] = np.where(
        (balls["balls_remaining"] > 0) & (balls["runs_required"] > 0),
        balls["runs_required"] * 6 / balls["balls_remaining"],
        0,
    )

    balls.loc[balls["inning"] == 0, "required_run_rate"] = 0
    balls.loc[balls["runs_required"] <= 0, "required_run_rate"] = 0

    print("Adding bowler type feature...")
    with open("../New Data/data/updated_pacers.json", "r") as f:
        pacers = json.load(f)

    balls["is_pacer"] = balls["bowler"].isin(pacers).astype(int)

    balls["over"] = balls["over"] / 20
    balls["sin_ball"] = np.sin(2 * np.pi * balls["legal_ball"] / 6)
    balls["cos_ball"] = np.cos(2 * np.pi * balls["legal_ball"] / 6)

    balls["rr_momentum"] = balls["required_run_rate"] - balls["current_run_rate"]

    print("Dropping columns...")
    balls.drop(
        columns=[
            "Byes",
            "LegByes",
            "Penalty",
            "ball",
            "balls_bowled",
            "batsman_runs",
            "batting_team",
            "bowling_team",
            "date",
            "isNoBall",
            "isWide",
            "is_legal",
            "is_wicket",
            "legal_ball",
            "legal_ball_1",
            "over_number",
            "overs_bowled",
            "player_dismissed",
            "runs_required",
            "total_runs",
            "wickets_fallen",
        ],
        inplace=True,
    )

    print("Normalization...")
    balls["balls_remaining"] /= 120
    balls["wickets_before"] /= 10
    balls["score_before"] /= 180
    balls["target"] /= 180
    balls["total_balls"] /= 10

    balls["current_score"] /= 180
    balls["current_run_rate"] /= 36
    balls["required_run_rate"] /= 36
    balls["required_run_rate"] = balls["required_run_rate"].clip(upper=2)
    balls["batter_history_matches"] /= 100
    balls["last_1_runs"] /= 100
    balls["last_2_runs"] /= 100
    balls["last_3_runs"] /= 100
    balls["last_1_balls"] /= 60
    balls["last_2_balls"] /= 60
    balls["last_3_balls"] /= 60

    balls["bowler_history_matches"] /= 100
    balls["last_1_runs_conceded"] /= 40
    balls["last_2_runs_conceded"] /= 40
    balls["last_3_runs_conceded"] /= 40
    balls["last_1_balls_bowled"] /= 24
    balls["last_2_balls_bowled"] /= 24
    balls["last_3_balls_bowled"] /= 24
    balls["rr_momentum"] /= 10
    balls['rr_momentum'] = balls['rr_momentum'].clip(-2, 2)

    print("Final shape:", balls.shape)

    print("Saving features...")
    balls.to_parquet(FEATURES_PATH, index=False)

    print("Features saved successfully.")


if __name__ == "__main__":
    build_features()
