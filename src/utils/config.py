"""
Central configuration — all env vars and constants in one place.
Never import os.getenv() directly in other modules; import from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (local development)
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_PATH    = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "lgbm_model.joblib"))
SQLITE_DB     = Path(os.getenv("SQLITE_DB_PATH", BASE_DIR / "sql" / "credit_risk.db"))
LOGS_DIR      = BASE_DIR / "logs"
MODELS_DIR    = BASE_DIR / "models"

# Create directories if missing
for _dir in [LOGS_DIR, MODELS_DIR, SQLITE_DB.parent]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# ── ML ────────────────────────────────────────────────────────────────────────
RANDOM_SEED      = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE        = 0.2
CV_FOLDS         = 5
SAMPLE_ROWS      = int(os.getenv("SAMPLE_ROWS", 50000))

# Risk thresholds
HIGH_RISK_THRESHOLD   = 0.60
MEDIUM_RISK_THRESHOLD = 0.35

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "application_train.csv"
TEST_FILE  = "application_test.csv"

# Columns to drop due to high missingness (>60%)
HIGH_MISSING_THRESHOLD = 0.60

# Target column
TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"
