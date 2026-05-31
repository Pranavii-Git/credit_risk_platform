"""
Data preprocessing pipeline.

Design principles:
  - Every transformation is documented with a business reason
  - No data leakage: fit only on training split
  - Reproducible: all random ops use RANDOM_SEED from config
  - Returns both the processed DataFrame AND a fitted pipeline artifact
    so inference uses identical transformations
"""

import warnings
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.utils.config import HIGH_MISSING_THRESHOLD, RANDOM_SEED, TARGET_COL, ID_COL
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Known anomaly fixes (domain knowledge, not statistics)
# ─────────────────────────────────────────────────────────────────────────────

def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix domain-specific anomalies that would mislead the model.

    DAYS_EMPLOYED = 365243 is a sentinel used by Home Credit to mark
    applicants with no employment record (retired, unemployed, etc.).
    Leaving it as-is gives the model a spurious 1000-year employment
    signal — we replace with NaN so imputation handles it naturally.
    """
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
        df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
        log.info(f"Fixed DAYS_EMPLOYED anomaly: {anomaly_count:,} rows → NaN")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Drop extremely sparse columns
# ─────────────────────────────────────────────────────────────────────────────

def drop_high_missing(df: pd.DataFrame, threshold: float = HIGH_MISSING_THRESHOLD) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns where >threshold fraction of values are missing.

    Business reason: imputing >60% missing data introduces more noise
    than signal and can mislead the model.
    """
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    # Never drop ID or target
    cols_to_drop = [c for c in cols_to_drop if c not in [ID_COL, TARGET_COL]]

    df = df.drop(columns=cols_to_drop)
    log.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
    return df, cols_to_drop


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features that improve model discrimination.

    Each feature has a clear credit-risk rationale:
    - CREDIT_INCOME_RATIO: Debt burden — higher = riskier
    - ANNUITY_INCOME_RATIO: Monthly repayment strain
    - CREDIT_TERM: Implied loan duration
    - DAYS_EMPLOYED_RATIO: Stability relative to age
    - AGE_YEARS: Human-readable age (positive)
    - INCOME_PER_FAMILY_MEMBER: Household affordability
    """
    df = df.copy()

    eps = 1e-6  # prevent division by zero

    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)

    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + eps)

    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365.25

    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_FAMILY"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)

    log.info("Feature engineering complete — added 6 derived features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Encode categoricals
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode all object columns.

    We use LabelEncoding (not OneHot) for tree-based models which handle
    ordinal splits natively.  LightGBM and XGBoost work well with this.
    The fitted encoders are saved so inference can use identical mappings.
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    log.info(f"Encoded {len(cat_cols)} categorical columns")
    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Impute remaining missing values
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute remaining NaN with column medians.

    Median is preferred over mean for financial data because it is
    robust to skewed distributions and outliers (e.g. extreme incomes).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Exclude target and id from imputation
    impute_cols = [c for c in num_cols if c not in [TARGET_COL, ID_COL]]

    imputer = SimpleImputer(strategy="median")
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    log.info(f"Imputed missing values in {len(impute_cols)} numeric columns (strategy=median)")
    return df, imputer


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Args:
        df: Raw DataFrame from loader.load_train()

    Returns:
        (processed_df, artifacts_dict)
        artifacts_dict contains encoders, imputer, dropped_cols — needed
        to reproduce the same transformation at inference time.
    """
    log.info("Starting preprocessing pipeline ...")

    df = fix_known_anomalies(df)
    df, dropped_cols = drop_high_missing(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    df, imputer = impute_missing(df)

    artifacts = {
        "dropped_cols": dropped_cols,
        "encoders": encoders,
        "imputer": imputer,
    }

    log.info(f"Preprocessing complete — final shape: {df.shape}")
    return df, artifacts


def preprocess_single_record(record: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply saved preprocessing artifacts to a single inference record.

    Args:
        record: Dict of feature values from UI input form.
        artifacts: Dict returned by preprocess() during training.

    Returns:
        Single-row DataFrame ready for model.predict_proba()
    """
    df = pd.DataFrame([record])

    # Fix anomaly
    df = fix_known_anomalies(df)

    # Drop same columns as training
    cols_to_drop = [c for c in artifacts.get("dropped_cols", []) if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Engineer same features
    df = engineer_features(df)

    # Encode using saved encoders
    for col, le in artifacts.get("encoders", {}).items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Impute using saved imputer
    imputer: SimpleImputer = artifacts.get("imputer")
    if imputer is not None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        available = [c for c in imputer.feature_names_in_ if c in df.columns]
        missing_from_record = [c for c in imputer.feature_names_in_ if c not in df.columns]
        for c in missing_from_record:
            df[c] = np.nan
        df[imputer.feature_names_in_] = imputer.transform(df[imputer.feature_names_in_])

    return df
