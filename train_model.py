"""
Standalone model training script.

Run this directly if you want to train outside of the Streamlit UI:
    python train_model.py

Or inside Docker:
    docker-compose exec credit-risk-platform python train_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.utils.config import DATA_DIR, MODEL_PATH, TRAIN_FILE

log = get_logger("train_model")


def main():
    log.info("=" * 60)
    log.info("Credit Risk Platform — Training Pipeline")
    log.info("=" * 60)

    # Step 1: Check dataset
    train_path = Path(DATA_DIR) / TRAIN_FILE
    if not train_path.exists():
        log.error(f"Dataset not found: {train_path}")
        log.error("Download application_train.csv from Kaggle and place it in ./data/")
        sys.exit(1)

    # Step 2: Load raw data
    log.info("Step 1/5 — Loading raw data ...")
    from src.data.loader import load_train
    df_raw = load_train()

    # Step 3: Preprocess
    log.info("Step 2/5 — Preprocessing ...")
    from src.data.preprocessor import preprocess
    df_processed, artifacts = preprocess(df_raw)

    # Step 4: Train all models + save bundle with metrics
    log.info("Step 3/5 — Training models (CV comparison) ...")
    from src.ml.train import run_training
    results = run_training(df_processed, artifacts)

    # Step 5: Print results
    log.info("Step 4/5 — Evaluation complete.")

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (Cross-Validation ROC-AUC)")
    print("=" * 60)
    print(results["cv_results"].to_string(index=False))

    metrics = results.get("test_metrics", {})
    if metrics:
        print("\n" + "=" * 60)
        print("  FINAL MODEL (LightGBM) — TEST SET RESULTS")
        print("=" * 60)
        print(f"  ROC-AUC  : {metrics.get('roc_auc', 'N/A')}")
        print(f"  PR-AUC   : {metrics.get('pr_auc', 'N/A')}")
        print(f"  F1 Score : {metrics.get('f1_score', 'N/A')}")
        print(f"  Precision: {metrics.get('precision', 'N/A')}")
        print(f"  Recall   : {metrics.get('recall', 'N/A')}")
        print(f"  Accuracy : {metrics.get('accuracy', 'N/A')}")
        print(f"  Threshold: {metrics.get('threshold', 'N/A')}")
        print(f"  Train rows: {metrics.get('train_rows', 'N/A'):,}")
        print(f"  Test rows : {metrics.get('test_rows', 'N/A'):,}")
        print("=" * 60)

    # Step 6: Create SQLite for chatbot
    log.info("Step 5/5 — Creating SQLite DB for chatbot ...")
    from src.data.loader import load_or_create_sqlite
    load_or_create_sqlite()

    print(f"\n✅ Training complete! Model saved to: {MODEL_PATH}")
    print("Run 'streamlit run app.py' to start the UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()