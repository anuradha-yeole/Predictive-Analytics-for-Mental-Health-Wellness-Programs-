import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from .config import RAW_DATA_PATH, PREPROCESSED_CSV_PATH, MODELS_DIR, LABEL_ENCODER_PATH


TARGET_COL = "Psychological Health"


def encode_target(series: pd.Series) -> (pd.Series, Dict[str, int]):
    """Map 'Good'/'Bad' → 1/0 (or whatever appears in the data)."""
    unique_vals = series.dropna().unique().tolist()
    mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
    encoded = series.map(mapping)
    return encoded, mapping


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH)

    # Strip column names of whitespace
    df.columns = [c.strip() for c in df.columns]

    # Drop completely duplicate rows if any
    df = df.drop_duplicates()

    # Handle missing income: median imputation
    if "Income" in df.columns:
        median_income = df["Income"].median()
        df["Income"] = df["Income"].fillna(median_income)

    # Handle missing psychosocial factors: mode imputation
    if "Psychosocial Factors" in df.columns:
        mode_psy = df["Psychosocial Factors"].mode()
        if not mode_psy.empty:
            df["Psychosocial Factors"] = df["Psychosocial Factors"].fillna(mode_psy.iloc[0])

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Encode target
    y_encoded, label_mapping = encode_target(df[TARGET_COL])
    df[TARGET_COL] = y_encoded

    # Optional: simple age bucket feature
    if "Age in years" in df.columns:
        bins = [0, 17, 25, 35, 50, 65, 120]
        labels = ["<18", "18-25", "26-35", "36-50", "51-65", "65+"]
        df["AgeBucket"] = pd.cut(df["Age in years"], bins=bins, labels=labels, right=True)

    # Create models dir if not exists
    Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)

    # Save preprocessed data
    PREPROCESSED_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(PREPROCESSED_CSV_PATH, index=False)

    # Save label mapping for later interpretation
    with open(LABEL_ENCODER_PATH, "w") as f:
        json.dump({"target_mapping": label_mapping}, f, indent=2)

    print(f"Preprocessed data saved to {PREPROCESSED_CSV_PATH}")
    print(f"Target mapping: {label_mapping}")


if __name__ == "__main__":
    main()
