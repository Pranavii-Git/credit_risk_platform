"""
Central configuration — all env vars and constants in one place.
Never import os.getenv() directly in other modules; import from here.
"""
import os
from pathlib import Path

# Load .env file if present (local development only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Base directory ────────────────────────────────────────────────────────────
# Works on local PC and Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Detect Streamlit Cloud ────────────────────────────────────────────────────
_ON_STREAMLIT_CLOUD = Path("/mount/src").exists()
_CLOUD_ROOT = Path("/mount/src/credit_risk_platform")

# ── Paths ─────────────────────────────────────────────────────────────────────
def _resolve(env_key: str, *relative_parts) -> Path:
    """
    Read path from env var if set, otherwise build from BASE_DIR.
    On Streamlit Cloud, fall back to /mount/src/credit_risk_platform if needed.
    """
    val = os.getenv(env_key)
    if val:
        return Path(val)
    local_path = BASE_DIR.joinpath(*relative_parts)
    if _ON_STREAMLIT_CLOUD:
        cloud_path = _CLOUD_ROOT.joinpath(*relative_parts)
        return cloud_path if cloud_path.exists() else local_path
    return local_path

DATA_DIR   = _resolve("DATA_DIR",       "data")
MODEL_PATH = _resolve("MODEL_PATH",     "models", "lgbm_model.joblib")
SQLITE_DB  = _resolve("SQLITE_DB_PATH", "sql",    "credit_risk.db")
LOGS_DIR   = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if missing (safe — won't crash on read-only filesystems)
for _dir in [LOGS_DIR, MODELS_DIR]:
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER",      "groq")
LLM_MODEL         = os.getenv("LLM_MODEL",         "llama-3.1-8b-instant")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",      "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

# ── ML ────────────────────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE   = 0.2
CV_FOLDS    = 5
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "50000"))

# Risk thresholds
HIGH_RISK_THRESHOLD   = 0.60
MEDIUM_RISK_THRESHOLD = 0.35

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_FILE = "application_train.csv"
TEST_FILE  = "application_test.csv"

HIGH_MISSING_THRESHOLD = 0.60
TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"
