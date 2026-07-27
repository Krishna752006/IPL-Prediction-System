import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class TabTransformerLSTM(nn.Module):

    def __init__(
        self,
        num_players,
        num_venues,
        num_seasons,
        numerical_dim,
        num_match_states=6,
        player_embedding_dim=60,
        venue_embedding_dim=30,
        season_embedding_dim=8,
        state_embedding_dim=8,
        transformer_dim=128,
        transformer_heads=8,
        transformer_layers=2,
        lstm_hidden_dim=128,
        lstm_layers=2,
        dropout=0.1,
        mlp_dropout=0.2,
    ):
        super().__init__()

        self.batter_embedding = nn.Embedding(
            num_players,
            player_embedding_dim,
            padding_idx=0,
        )

        self.non_striker_embedding = nn.Embedding(
            num_players,
            player_embedding_dim,
            padding_idx=0,
        )

        self.bowler_embedding = nn.Embedding(
            num_players,
            player_embedding_dim,
            padding_idx=0,
        )

        self.venue_embedding = nn.Embedding(
            num_venues,
            venue_embedding_dim,
            padding_idx=0,
        )

        self.season_embedding = nn.Embedding(
            num_seasons + 1,
            season_embedding_dim,
            padding_idx=0,
        )

        self.match_state_embedding = nn.Embedding(
            num_match_states, state_embedding_dim, padding_idx=0
        )

        categorical_dim = (
            player_embedding_dim * 3
            + venue_embedding_dim
            + season_embedding_dim
            + state_embedding_dim
        )

        self.embedding_projection = nn.Linear(
            categorical_dim,
            transformer_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dropout=mlp_dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.pos_encoder = PositionalEncoding(transformer_dim)
        self.numerical_norm = nn.LayerNorm(numerical_dim)

        lstm_input_dim = transformer_dim + numerical_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.wicket_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(
                lstm_hidden_dim,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
        )

        self.score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(32, 1),
        )

        self.wicket_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(16, 1),
        )

        self.wide_head = nn.Sequential(
            nn.Linear(128, 48), nn.ReLU(), nn.Dropout(mlp_dropout), nn.Linear(48, 1)
        )

    def encode(self, numerical_features, categorical_features):
        batter_ids = categorical_features[:, :, 0]
        non_striker_ids = categorical_features[:, :, 1]
        bowler_ids = categorical_features[:, :, 2]
        venue_ids = categorical_features[:, :, 3]
        season_ids = categorical_features[:, :, 4]
        match_state_ids = categorical_features[:, :, 5]
        pad_mask = (batter_ids == 0).to(categorical_features.device)

        batter_embed = self.batter_embedding(batter_ids)
        non_striker_embed = self.non_striker_embedding(non_striker_ids)
        bowler_embed = self.bowler_embedding(bowler_ids)
        venue_embed = self.venue_embedding(venue_ids)
        season_embed = self.season_embedding(season_ids)
        state_embed = self.match_state_embedding(match_state_ids)

        categorical_embeddings = torch.cat(
            [
                batter_embed,
                non_striker_embed,
                bowler_embed,
                venue_embed,
                season_embed,
                state_embed,
            ],
            dim=-1,
        )

        projected = self.embedding_projection(categorical_embeddings)
        projected = self.pos_encoder(projected)
        transformed = self.transformer(
            projected, src_key_padding_mask=pad_mask
        )  # learned categorical embedding
        numerical_normed = self.numerical_norm(
            numerical_features
        )  # learned numerical embedding

        combined = torch.cat([transformed, numerical_normed], dim=-1)
        return transformed, numerical_normed, combined

    def forward(self, numerical_features, categorical_features):
        transformed, numerical_normed, combined = self.encode(
            numerical_features, categorical_features
        )

        lstm_output, _ = self.lstm(combined)
        final_hidden_score = lstm_output[:, -1, :]
        shared_features = self.shared_mlp(final_hidden_score)

        wicket_lstm_output, _ = self.wicket_lstm(combined)
        final_hidden_wicket = wicket_lstm_output[:, -1, :]

        score_pred = self.score_head(shared_features).squeeze(-1)
        wicket_logits = self.wicket_head(final_hidden_wicket).squeeze(-1)
        wide_logits = self.wide_head(shared_features).squeeze(-1)

        return {"score": score_pred, "wicket": wicket_logits, "wide": wide_logits}
