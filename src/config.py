from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "casestudy.xlsx"
PREPROCESSED_CSV_PATH = DATA_DIR / "preprocessed.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.json"
