"""
Model training pipeline.

Strategy:
  - Compare 3 models: Logistic Regression, XGBoost, LightGBM
  - Use Stratified K-Fold CV to handle class imbalance in evaluation
  - Final model = LightGBM (best AUC on this dataset class)
  - Class imbalance handled with class_weight='balanced' + threshold tuning
  - Save model + artifacts with joblib for reproducible inference
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.utils.config import (
    RANDOM_SEED, TEST_SIZE, CV_FOLDS, MODEL_PATH,
    TARGET_COL, ID_COL, MODELS_DIR
)
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_model_candidates() -> Dict[str, Any]:
    """
    Return the three model candidates with their configurations.

    Rationale for each:
    - LogisticRegression: Baseline. Interpretable coefficients. Fast.
      Weakness: assumes linear decision boundary.
    - XGBoost: Strong gradient boosting. Handles mixed types well.
      scale_pos_weight compensates for imbalance (neg/pos ratio ≈ 11).
    - LightGBM: Selected final model. Faster than XGBoost on large data,
      handles categoricals natively, leaf-wise growth gives better AUC
      at equivalent depth. DART dropout reduces overfitting.
    """
    neg_pos_ratio = 11  # approximate from EDA: 92% neg / 8% pos

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="saga",
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=neg_pos_ratio,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            boosting_type="dart",
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix and target vector, dropping id column."""
    drop_cols = [c for c in [TARGET_COL, ID_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return X, y


def cross_validate_all(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Run Stratified K-Fold CV on all candidate models.

    Stratified ensures each fold preserves the minority class ratio,
    which is critical for imbalanced data.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for name, model in models.items():
        log.info(f"Cross-validating {name} ...")
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append({
            "Model": name,
            "Mean AUC": round(auc_scores.mean(), 4),
            "Std AUC": round(auc_scores.std(), 4),
            "Min AUC": round(auc_scores.min(), 4),
            "Max AUC": round(auc_scores.max(), 4),
        })
        log.info(f"  {name}: AUC = {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    return pd.DataFrame(results).sort_values("Mean AUC", ascending=False)


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LGBMClassifier:
    """
    Train the final LightGBM model on the full training split.
    """
    log.info("Training final LightGBM model ...")
    model = get_model_candidates()["LightGBM"]
    model.fit(X_train, y_train)
    log.info("Final model training complete.")
    return model


def find_optimal_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Find the probability threshold that maximises F1 score on validation set.

    Default 0.5 threshold is sub-optimal for imbalanced data.
    We sweep thresholds and select the one with highest macro-F1.
    """
    from sklearn.metrics import f1_score

    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [
        f1_score(y_val, (probs >= t).astype(int), average="macro")
        for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(f1_scores)]
    log.info(f"Optimal threshold: {best_threshold:.2f} (F1={max(f1_scores):.4f})")
    return float(best_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Master training entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_training(df: pd.DataFrame, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        df: Preprocessed DataFrame (output of preprocessor.preprocess).
        artifacts: Preprocessing artifacts (encoders, imputer, etc.)

    Returns:
        training_results dict with model, metrics, cv_comparison, threshold.
    """
    log.info("=== Training Pipeline Started ===")

    X, y = split_features_target(df)
    log.info(f"Feature matrix: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Train / test split — stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Cross-validate all candidates
    models = get_model_candidates()
    cv_results = cross_validate_all(models, X_train, y_train)

    # Train final model
    final_model = train_final_model(X_train, y_train)

    # Optimal threshold
    threshold = find_optimal_threshold(final_model, X_test, y_test)

    # Save model + all artifacts
    save_bundle = {
        "model": final_model,
        "threshold": threshold,
        "feature_names": X_train.columns.tolist(),
        "preprocessing_artifacts": artifacts,
    }
    joblib.dump(save_bundle, MODEL_PATH)
    log.info(f"Model bundle saved → {MODEL_PATH}")

    return {
        "model": final_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv_results": cv_results,
        "threshold": threshold,
        "feature_names": X_train.columns.tolist(),
    }
