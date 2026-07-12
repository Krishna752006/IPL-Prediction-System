import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IPLDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        target_year: int,
        player_mapping: dict,
        venue_mapping: dict,
        mode: str = "train",
        sequence_length: int = 30,
    ):
        self.sequence_length = sequence_length
        self.mode = mode
        self.target_year = target_year

        if mode == "train":
            df = df[df["season"] < target_year].copy()

        elif mode == "embed":
            df = df[df["season"] == target_year].copy()

        elif mode in ["val", "test"]:
            season_df = df[df["season"] == target_year].copy()

            unique_matches = (
                season_df["matchId"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            split_idx = len(unique_matches) // 2

            val_matches = unique_matches[:split_idx]
            test_matches = unique_matches[split_idx:]

            if mode == "val":
                df = season_df[
                    season_df["matchId"].isin(val_matches)
                ].copy()

            else:
                df = season_df[
                    season_df["matchId"].isin(test_matches)
                ].copy()

        else:
            raise ValueError(
                "mode must be one of ['train', 'val', 'test']"
            )

        self.df = df.copy()

        df = df.sort_values(
            by=["matchId", "inning", "over", "total_balls"]
        ).reset_index(drop=True)

        self.player2idx = player_mapping
        self.venue2idx = venue_mapping

        df["batter_id"] = df["batsman"].map(self.player2idx)
        df["non_striker_id"] = df["non_striker"].map(self.player2idx)
        df["bowler_id"] = df["bowler"].map(self.player2idx)
        df["venue_id"] = df["venue"].map(self.venue2idx)
        df["season_id"] = (df["season"] - 2007 + 1).astype(int)

        self.feature_columns = [
            "inning",
            "over",
            "total_balls",
            "balls_remaining",
            "phase_pp",
            "phase_middle",
            "phase_death",
            "target",
            "is_pacer",
            # "score_before",
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

        self.categorical_columns = [
            "batter_id",
            "non_striker_id",
            "bowler_id",
            "venue_id",
            "season_id",
            "match_state_id",
        ]

        self.numerical_dim = len(self.feature_columns)
        self.categorical_dim = len(self.categorical_columns)

        self.num_players = len(self.player2idx) + 1
        self.num_venues = len(self.venue2idx) + 1
        self.num_seasons = int(df["season_id"].max()) + 1

        self.X_numerical = []
        self.X_categorical = []
        self.y_score = []
        self.y_wide = []
        self.y_wicket = []

        grouped = df.groupby(["matchId", "inning"])

        for (_, _), group in grouped:
            group = group.reset_index(drop=True)

            numerical_features = group[self.feature_columns].values
            categorical_features = group[self.categorical_columns].values
            score_targets = group["current_score"].values
            wide_targets = group["isWide_target"].values
            wicket_targets = group["is_wicket_target"].values

            for idx in range(len(group)):

                start_idx = max(0, idx - self.sequence_length + 1)

                numerical_seq = numerical_features[start_idx : idx + 1]
                categorical_seq = categorical_features[start_idx : idx + 1]

                pad_size = self.sequence_length - len(numerical_seq)

                if pad_size > 0:
                    numerical_padding = np.zeros(
                        (pad_size, len(self.feature_columns)),
                        dtype=np.float32,
                    )

                    categorical_padding = np.zeros(
                        (pad_size, len(self.categorical_columns)),
                        dtype=np.int64,
                    )

                    numerical_seq = np.vstack([numerical_padding, numerical_seq])

                    categorical_seq = np.vstack([categorical_padding, categorical_seq])

                self.X_numerical.append(numerical_seq)
                self.X_categorical.append(categorical_seq)
                self.y_score.append(score_targets[idx])
                self.y_wide.append(wide_targets[idx])
                self.y_wicket.append(wicket_targets[idx])

        self.X_numerical = np.array(self.X_numerical, dtype=np.float32)
        self.X_categorical = np.array(self.X_categorical, dtype=np.int64)
        self.y_score = np.array(self.y_score, dtype=np.float32)
        self.y_wide = np.array(self.y_wide, dtype=np.float32)
        self.y_wicket = np.array(self.y_wicket, dtype=np.float32)
    
        print("Dataset Built Successfully")
        print(f"Samples: {len(self.y_score)}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Numerical Shape: {self.X_numerical.shape}")
        print(f"Categorical Shape: {self.X_categorical.shape}")

    def __len__(self):
        return len(self.y_score)

    def __getitem__(self, idx):

        numerical_tensor = torch.tensor(
            self.X_numerical[idx],
            dtype=torch.float32,
        )

        categorical_tensor = torch.tensor(
            self.X_categorical[idx],
            dtype=torch.long,
        )

        score_target = torch.tensor(
            self.y_score[idx],
            dtype=torch.float32,
        )

        wide_target = torch.tensor(
            self.y_wide[idx],
            dtype=torch.float32,
        )

        wicket_target = torch.tensor(
            self.y_wicket[idx],
            dtype=torch.float32,
        )

        return {
            "numerical_features": numerical_tensor,
            "categorical_features": categorical_tensor,
            "score_target": score_target,
            "wide_target": wide_target,
            "wicket_target": wicket_target,
        }
