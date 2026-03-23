import os

import config
import numpy as np
import pandas as pd
from features import build_features
from ingest import load_data
from split import split_data


def run_pipeline():
    # 1. Load
    df = load_data(config.RAW_PATH)

    # 2. Features
    X, y, encoder, scalerx, scalery = build_features(
        df, config.INPUT_FEATURES, config.X_FEATS, config.Y_FEATS
    )

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Save processed dataset (simple version)
    os.makedirs(os.path.dirname(config.OUTPUT_PATH), exist_ok=True)

    df_out = pd.DataFrame(X)
    df_out["y_runs"] = y[:, 0]
    df_out["y_wickets"] = y[:, 1]

    df_out.to_parquet(config.OUTPUT_PATH)

    print("Pipeline completed")
    print("Saved to:", config.OUTPUT_PATH)
    print("Shape:", df_out.shape)


if __name__ == "__main__":
    run_pipeline()
