from pathlib import Path

DATASET_VERSION = "ab"

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_MATCHES = BASE_DIR / "New Data/matches_updated_ipl_upto_2025.csv"
RAW_DELIVERIES = BASE_DIR / "New Data/deliveries_updated_ipl_upto_2025.csv"

PROCESSED_DIR = BASE_DIR / "ml-service/data/processed"

VERSION_DIR = PROCESSED_DIR / DATASET_VERSION

CLEAN_MATCHES_PATH = VERSION_DIR / "clean_matches.parquet"
CLEAN_DELIVERIES_PATH = VERSION_DIR / "clean_deliveries.parquet"

OFFLINE_STATS_DIR = VERSION_DIR / "offline_stats"

VENUE_PHASE_AVG_PATH = OFFLINE_STATS_DIR / "venue_phase_averages.json"
BATTER_FORM_PATH = OFFLINE_STATS_DIR / "recent_form_batter.parquet"
BOWLER_FORM_PATH = OFFLINE_STATS_DIR / "recent_form_bowler.parquet"
