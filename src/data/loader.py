"""
Dataset loading module.

Responsibilities:
  - Load raw CSV files from DATA_DIR
  - Create / refresh the SQLite database used by the NL-to-SQL chatbot
  - Provide a lightweight cached sample for UI responsiveness

All paths come from config so Docker volume mounts work transparently.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import DATA_DIR, SQLITE_DB, SAMPLE_ROWS, TARGET_COL, ID_COL, TRAIN_FILE
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_train(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load application_train.csv.

    Args:
        nrows: If set, load only the first N rows (useful for quick testing).

    Returns:
        Raw DataFrame.
    """
    path = Path(DATA_DIR) / TRAIN_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found at {path}.\n"
            "Place application_train.csv inside the ./data/ directory."
        )

    log.info(f"Loading training data from {path} ...")
    df = pd.read_csv(path, nrows=nrows)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def load_or_create_sqlite(force_refresh: bool = False) -> str:
    """
    Ensure the SQLite DB exists and return its path string.

    We store a random sample (SAMPLE_ROWS) of application_train in an
    'applications' table.  The full dataset rows are unnecessary for the
    chatbot and would make every query slow.

    Args:
        force_refresh: Drop and re-create the table even if it already exists.

    Returns:
        Absolute path to the SQLite file.
    """
    db_path = str(SQLITE_DB)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table already populated
    table_exists = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='applications'"
    ).fetchone()

    if table_exists and not force_refresh:
        count = cursor.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        log.info(f"SQLite DB already has {count:,} rows — skipping reload.")
        conn.close()
        return db_path

    log.info(f"Building SQLite DB at {db_path} (sample = {SAMPLE_ROWS:,} rows) ...")
    df = load_train()

    # Random sample — stratify to keep TARGET distribution
    if len(df) > SAMPLE_ROWS:
        df = df.groupby(TARGET_COL, group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_ROWS / len(df), random_state=42)
        )

    df.to_sql("applications", conn, if_exists="replace", index=False)
    conn.close()

    log.info(f"SQLite DB ready — {len(df):,} rows written.")
    return db_path


def get_schema_string() -> str:
    """
    Return a human-readable schema description of the applications table.
    Used in LLM prompts for schema grounding.
    """
    return """
Table: applications
Description: Home Credit loan application records. Each row is one applicant.

Key columns:
  SK_ID_CURR           INTEGER  — Unique applicant ID
  TARGET               INTEGER  — 1 = defaulted, 0 = repaid (label)
  AMT_CREDIT           REAL     — Total loan amount requested (USD)
  AMT_INCOME_TOTAL     REAL     — Applicant annual income (USD)
  AMT_ANNUITY          REAL     — Monthly loan repayment amount (USD)
  AMT_GOODS_PRICE      REAL     — Price of goods/property financed
  NAME_CONTRACT_TYPE   TEXT     — 'Cash loans' or 'Revolving loans'
  CODE_GENDER          TEXT     — 'M' (male) or 'F' (female)
  FLAG_OWN_CAR         TEXT     — 'Y' or 'N' — owns a car
  FLAG_OWN_REALTY      TEXT     — 'Y' or 'N' — owns property
  CNT_CHILDREN         INTEGER  — Number of children
  NAME_INCOME_TYPE     TEXT     — Employment type (e.g. 'Working', 'Pensioner')
  NAME_EDUCATION_TYPE  TEXT     — Highest education level
  NAME_FAMILY_STATUS   TEXT     — Marital status
  NAME_HOUSING_TYPE    TEXT     — Housing situation
  DAYS_BIRTH           INTEGER  — Age in days (NEGATIVE — divide by -365 for years)
  DAYS_EMPLOYED        INTEGER  — Employment duration in days (NEGATIVE, or 365243 = N/A)
  OCCUPATION_TYPE      TEXT     — Job type
  CNT_FAM_MEMBERS      REAL     — Family size
  REGION_RATING_CLIENT INTEGER  — Region risk rating (1=low, 3=high)
  EXT_SOURCE_1         REAL     — External credit score 1 (0–1, higher = better)
  EXT_SOURCE_2         REAL     — External credit score 2 (0–1, higher = better)
  EXT_SOURCE_3         REAL     — External credit score 3 (0–1, higher = better)
  CREDIT_INCOME_RATIO  REAL     — Engineered: AMT_CREDIT / AMT_INCOME_TOTAL
  ANNUITY_INCOME_RATIO REAL     — Engineered: AMT_ANNUITY / AMT_INCOME_TOTAL

SQLite dialect notes:
  - Age in years: CAST(DAYS_BIRTH AS REAL) / -365.0
  - DAYS_EMPLOYED = 365243 means unemployed/not applicable
  - All monetary values are in the same currency unit
""".strip()
