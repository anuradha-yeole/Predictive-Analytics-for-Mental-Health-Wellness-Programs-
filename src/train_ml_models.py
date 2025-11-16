import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from .config import PREPROCESSED_CSV_PATH, BEST_MODEL_PATH, MODELS_DIR, LABEL_ENCODER_PATH, TARGET_COL


def load_data():
    df = pd.read_csv(PREPROCESSED_CSV_PATH)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Identify numerical vs categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def build_models():
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    return {"random_forest": rf, "xgboost": xgb_clf}


def evaluate_model(pipe, X_test, y_test, name: str):
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("ROC AUC:", roc)
    print(classification_report(y_test, y_pred))

    return acc, roc


def compute_shap(pipe, X_train, numeric_cols, categorical_cols):
    """
    Compute SHAP values for the XGBoost model inside the pipeline.
    """
    preprocessor: ColumnTransformer = pipe.named_steps["preprocessor"]
    model: xgb.XGBClassifier = pipe.named_steps["model"]

    # Sample some rows for SHAP
    X_sample = X_train.sample(min(2000, len(X_train)), random_state=42)
    X_trans = preprocessor.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Get post-encoding feature names
    num_features = numeric_cols
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    feature_names = num_features + cat_features

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=True)


def main():
    X, y, numeric_cols, categorical_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models()

    best_score = -np.inf
    best_name = None
    best_pipe = None

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        acc, roc = evaluate_model(pipe, X_test, y_test, name)

        if roc > best_score:
            best_score = roc
            best_name = name
            best_pipe = pipe

    Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)
    joblib.dump(best_pipe, BEST_MODEL_PATH)
    print(f"\nSaved best model ({best_name}) with ROC AUC={best_score:.4f} at {BEST_MODEL_PATH}")

    # Compute SHAP only for XGBoost if it won
    if isinstance(best_pipe.named_steps["model"], xgb.XGBClassifier):
        print("\nComputing SHAP values for XGBoost...")
        compute_shap(best_pipe, X_train, numeric_cols, categorical_cols)
    else:
        print("\nBest model is not XGBoost, skipping SHAP (or adapt as needed).")


if __name__ == "__main__":
    main()
