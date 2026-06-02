# Credit Risk Intelligence Platform

An AI-powered end-to-end credit risk assessment system built on the Home Credit Default Risk dataset. Combines machine learning, explainable AI, and a natural language data interface into a single deployable application.

---

## Live Demo

**Deployed App:** https://creditriskplatformgit-fdegsjctmuk3dme6ozdfri.streamlit.app/

Note: EDA tab requires local dataset. All other tabs work fully on the live demo.

---

## Problem Statement

Banks need to make fast, accurate, and explainable credit decisions. Manual underwriting is slow and inconsistent. This platform addresses that gap by:

- Predicting loan default probability using a calibrated LightGBM model
- Explaining each prediction with SHAP feature attributions
- Letting business analysts query applicant data in plain English
- Providing auditable decision rules for regulatory compliance
- Delivering everything through a single-click Docker deployment

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                Credit Risk Intelligence Platform                  |
|                                                                    |
|  +-----------+   +-------------+   +-------------------------+   |
|  | CSV Data  |   |  SQLite DB  |   |   LLM (Groq Free)       |   |
|  | (307K rows|   |  (50K rows) |   |   NL to SQL Generation  |   |
|  +-----+-----+   +------+------+   +-------------+-----------+   |
|        |                |                         |               |
|  +-----+-----+   +------+------+   +-------------+-----------+   |
|  | loader.py |   |query_runner |   |     nl_to_sql.py        |   |
|  | preprocess|   |   .py       |   |  4-Layer Hallucination  |   |
|  |    or.py  |   |             |   |  Reduction Pipeline     |   |
|  +-----+-----+   +------+------+   +-------------+-----------+   |
|        |                |                         |               |
|  +-----+-----+          |                         |               |
|  | train.py  |          +-------------------------+               |
|  | predict.py|                      |                             |
|  | evaluate  |                      |                             |
|  +-----+-----+                      |                             |
|        +------------------------------+                           |
|                    Streamlit UI (app.py)                          |
|         EDA - Risk Prediction - Chat - Insights - Overview        |
+------------------------------------------------------------------+
```

### Component Diagram

| Component | File | Responsibility |
|-----------|------|----------------|
| Data Loader | `src/data/loader.py` | CSV loading, SQLite creation |
| Preprocessor | `src/data/preprocessor.py` | Cleaning, encoding, imputation, feature engineering |
| Training | `src/ml/train.py` | 3-model comparison, CV, final LightGBM training |
| Inference | `src/ml/predict.py` | Risk scoring, SHAP explanations |
| Evaluation | `src/ml/evaluate.py` | Metrics, Plotly charts |
| NL-to-SQL | `src/talk_to_data/nl_to_sql.py` | Full 4-step pipeline |
| SQL Runner | `src/talk_to_data/query_runner.py` | Safe SQLite execution |
| Prompts | `src/talk_to_data/prompt_templates.py` | Versioned LLM prompts |
| EDA | `notebooks/eda.py` | 11 EDA charts, 10 business insights |
| UI | `app.py` | Streamlit 5-tab interface |

---

## Quick Run (No Training Needed)

The trained model and SQLite database are included in the repository. No training required.

```bash
# 1. Clone
git clone https://github.com/Pranavii-Git/credit_risk_platform
cd credit_risk_platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 4. Run
streamlit run app.py
```

Open **http://localhost:8501**

Tabs 2 to 5 work immediately. Tab 1 (EDA) requires the dataset — see below.

---

## To Enable EDA Tab

Download `application_train.csv` from Kaggle and place it in `./data/`:

```
https://www.kaggle.com/competitions/home-credit-default-risk/data
```

Then refresh the app. The EDA Dashboard will load automatically.

---

## To Train From Scratch

1. Place `application_train.csv` in `./data/`
2. Run `streamlit run app.py`
3. Click **Train Model Now** in the sidebar
4. Wait 5-10 minutes for training to complete

---

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Project Structure

```
credit_risk_platform/
├── data/                          <- Place CSV files here (not committed)
├── documents/
│   └── project_presentation.pdf  <- Submission presentation
├── notebooks/
│   └── eda.py                     <- EDA charts and insights module
├── src/
│   ├── data/
│   │   ├── loader.py              <- CSV + SQLite loading
│   │   └── preprocessor.py        <- Full preprocessing pipeline
│   ├── ml/
│   │   ├── train.py               <- 3-model training + CV
│   │   ├── predict.py             <- Inference + SHAP explanations
│   │   └── evaluate.py            <- ROC-AUC, PR curve, confusion matrix
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py           <- NL-to-SQL orchestrator
│   │   ├── query_runner.py        <- Safe SQL execution
│   │   └── prompt_templates.py    <- Versioned prompts (v1.2)
│   └── utils/
│       ├── config.py              <- All env vars / constants
│       ├── logger.py              <- Loguru logging
│       └── helpers.py             <- Risk banding, formatting
├── sql/
│   ├── schema.sql                 <- SQLite schema reference
│   └── credit_risk.db             <- Pre-built SQLite database (committed)
├── models/
│   └── lgbm_model.joblib          <- Pre-trained model (committed)
├── app.py                         <- Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Groq API key (free at console.groq.com) |
| `LLM_PROVIDER` | No | `groq` | LLM provider |
| `LLM_MODEL` | No | `llama-3.1-8b-instant` | LLM model name |
| `DATA_DIR` | No | `./data` | Path to CSV files |
| `MODEL_PATH` | No | `./models/lgbm_model.joblib` | Path to trained model |
| `SQLITE_DB_PATH` | No | `./sql/credit_risk.db` | Path to SQLite database |
| `SAMPLE_ROWS` | No | `50000` | Rows loaded into SQLite |
| `RANDOM_SEED` | No | `42` | Reproducibility seed |

---

## EDA Key Insights

1. **External credit scores dominate** — EXT_SOURCE_2 and EXT_SOURCE_3 are the top predictors. Applicants with EXT_SOURCE_2 < 0.4 default at nearly 3x the baseline rate.

2. **8% class imbalance** — Only 8.07% of applicants defaulted. Training without balancing makes accuracy misleading. ROC-AUC is the right metric here.

3. **DAYS_EMPLOYED sentinel** — 55,374 rows (18% of data) have DAYS_EMPLOYED = 365,243, a placeholder for "not employed." This must be replaced with NaN before modelling.

4. **Age and risk** — Applicants aged 20-30 default at ~11% vs ~5% for those over 60. Youth correlates with risk.

5. **Education inversely predicts default** — Higher education applicants default at 5.3%; lower secondary at 10.6% — a 2x gap meaningful for scoring.

---

## Model Selection

Three models were compared using 5-fold Stratified CV:

| Model | CV ROC-AUC | Training Time | Selected |
|-------|------------|---------------|----------|
| Logistic Regression | 0.5922 +/- 0.0058 | ~11 min | No |
| XGBoost | 0.7606 +/- 0.0007 | ~1 min | No |
| LightGBM | 0.7580 +/- 0.0017 | ~3 min | Yes |

LightGBM was selected because:
- Best balance of AUC and training speed
- Leaf-wise growth outperforms depth-wise on tabular financial data
- DART boosting reduces overfitting without regularisation overhead
- Native handling of missing values — critical for this dataset's 24% missing rate
- SHAP compatible for full explainability

Class imbalance handled with `class_weight='balanced'` and optimal threshold tuning via F1 sweep (threshold = 0.69).

---

## Evaluation Metrics

| Metric | Value | Business Meaning |
|--------|-------|-----------------|
| ROC-AUC | 0.786 | Model ranks risky applicants above safe ones correctly 78.6% of the time |
| PR-AUC | 0.258 | Precision-recall balance for the minority default class |
| Precision | 0.286 | Of flagged high-risk applicants |
| Recall | 0.346 | Defaults caught by the model |
| F1 | 0.313 | Balanced precision-recall score |
| Threshold | 0.69 | Optimised for F1, not naive 0.5 |

---

## Hallucination Reduction Framework

4-layer defence approach:

| Layer | Technique | What It Prevents |
|-------|-----------|-----------------|
| 1 | Schema Grounding | Hallucinated column names |
| 2 | Rule-based Blocklist | SQL injection (INSERT/DELETE/DROP) |
| 3 | LLM Validation + Correction | Schema violations, syntax errors |
| 4 | Runtime Error Correction | Execution failures |

No RAG needed — the schema has ~25 key columns, fits in one prompt under 500 tokens.

---

## NL-to-SQL Workflow

```
User Question
     |
     v
[1] generate_sql()        <- Schema-grounded prompt -> LLM -> raw SQL
     |
     v
[2] validate_sql()        <- Rule blocklist check + LLM schema validation
     |
     v
[3] run_query()           <- Safe SQLite execution, retry on error
     |
     v
[4] generate_explanation() <- Result summary -> LLM -> business insight
     |
     v
UI: SQL shown + Data table + Explanation
```

Demo queries verified working:
1. Default rate by education level
2. Average loan amount by contract type
3. Income type with highest default rate
4. Top 10 highest-credit defaulting applicants
5. Default rate by gender

---

## Known Limitations

- Single-table NL-to-SQL: The chatbot only queries the applications table. Multi-table joins are not supported in this version.
- Training time: Training takes 5-10 minutes depending on hardware. A pre-trained model is committed to avoid this.
- LLM dependency: The chatbot requires an active Groq API key. Offline mode is not supported.
- EDA tab: Requires the 307MB dataset locally — cannot be hosted on Streamlit Cloud.

---

## Future Improvements

1. Multi-table joins — Load bureau.csv and previous_application.csv and extend the NL-to-SQL schema
2. RAG for policy documents — Embed credit policy PDFs for regulatory Q&A
3. Model retraining pipeline — Automated drift detection and scheduled retraining
4. Batch scoring API — REST endpoint using FastAPI for core banking integration
5. A/B testing framework — Compare prompt versions with automated accuracy metrics
6. Model calibration — Platt scaling for regulatory compliance

---

## Dataset

Home Credit Default Risk — Kaggle
https://www.kaggle.com/competitions/home-credit-default-risk/data

Primary file: `application_train.csv` (307,511 rows, 122 columns)
Target: `TARGET` (1 = defaulted, 0 = repaid) — 8.07% positive rate
