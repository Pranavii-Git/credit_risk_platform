# Credit Risk Intelligence Platform 🏦

An AI-powered end-to-end credit risk assessment system built on the Home Credit Default Risk dataset. Combines machine learning, explainable AI, and a natural language data interface into a single deployable application.

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
┌──────────────────────────────────────────────────────────────────┐
│                Credit Risk Intelligence Platform                  │
│                                                                    │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────────────────┐ │
│  │ CSV Data   │   │  SQLite DB  │   │   LLM (Claude / OpenAI)  │ │
│  │ (688 MB)   │   │  (50K rows) │   │   NL → SQL Generation    │ │
│  └─────┬──────┘   └──────┬──────┘   └─────────────┬────────────┘ │
│        │                 │                         │              │
│  ┌─────▼──────┐   ┌──────▼──────┐   ┌─────────────▼────────────┐ │
│  │ loader.py  │   │query_runner │   │     nl_to_sql.py          │ │
│  │ preprocess │   │   .py       │   │  4-Layer Hallucination    │ │
│  │    or.py   │   │             │   │  Reduction Pipeline       │ │
│  └─────┬──────┘   └──────┬──────┘   └─────────────┬────────────┘ │
│        │                 │                         │              │
│  ┌─────▼──────┐          │                         │              │
│  │ train.py   │          │                         │              │
│  │ predict.py │          └──────────────────────────┘             │
│  │ evaluate.py│                        │                          │
│  └─────┬──────┘                        │                          │
│        │                               │                          │
│        └───────────────────────────────▼──────────────────────┐  │
│                         Streamlit UI (app.py)                  │  │
│          Tab1: EDA · Tab2: ML · Tab3: Chat · Tab4: Insights    │  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
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

## Quick Start (3 Commands)

```bash
# 1. Clone and configure
git clone https://github.com/Pranavii-Git/credit_risk_platform
cd credit_risk_platform
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 2. Add the dataset
# Download application_train.csv from:
# https://www.kaggle.com/competitions/home-credit-default-risk/data
# Place it in ./data/application_train.csv

# 3. Run
docker-compose up --build
```

Open **http://localhost:8501** — the app will guide you through training the model via the sidebar.

---

## Step-by-Step Setup (without Docker)

```bash
# Python 3.10+ required
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your API key and set DATA_DIR=./data

# Place application_train.csv in ./data/

streamlit run app.py
```

---

## Project Structure

```
credit_risk_platform/
├── data/                          ← Place CSV files here (not committed)
├── documents/
│   └── project_presentation.pdf  ← Submission presentation
├── notebooks/
│   └── eda.py                     ← EDA charts and insights module
├── src/
│   ├── data/
│   │   ├── loader.py              ← CSV + SQLite loading
│   │   └── preprocessor.py        ← Full preprocessing pipeline
│   ├── ml/
│   │   ├── train.py               ← 3-model training + CV
│   │   ├── predict.py             ← Inference + SHAP explanations
│   │   └── evaluate.py            ← ROC-AUC, PR curve, confusion matrix
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py           ← NL-to-SQL orchestrator
│   │   ├── query_runner.py        ← Safe SQL execution
│   │   └── prompt_templates.py    ← Versioned prompts (v1.2)
│   └── utils/
│       ├── config.py              ← All env vars / constants
│       ├── logger.py              ← Loguru logging
│       └── helpers.py             ← Risk banding, formatting
├── sql/
│   └── schema.sql                 ← SQLite schema reference
├── models/                        ← Saved model artifacts (auto-generated)
├── app.py                         ← Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## EDA Key Insights

1. **External credit scores dominate** — EXT_SOURCE_2 and EXT_SOURCE_3 are the top predictors. Applicants with EXT_SOURCE_2 < 0.4 default at nearly 3× the baseline rate.

2. **8% class imbalance** — Only 8.07% of applicants defaulted. Training without balancing makes accuracy misleading. ROC-AUC is the right metric here.

3. **DAYS_EMPLOYED sentinel** — 55,374 rows (18% of data) have DAYS_EMPLOYED = 365,243, a placeholder for "not employed." This must be replaced with NaN before modelling.

4. **Age and risk** — Applicants aged 20–30 default at ~11% vs ~5% for those over 60. Youth correlates with risk.

5. **Education inversely predicts default** — Higher education applicants default at 5.3%; lower secondary at 10.6% — a 2× gap meaningful for scoring.

---

## Model Selection Rationale

Three models were compared using 5-fold Stratified CV:

| Model | CV ROC-AUC | Training Time | Selected |
|-------|------------|---------------|----------|
| Logistic Regression | ~0.716 ± 0.003 | <30s | ❌ |
| XGBoost | ~0.748 ± 0.004 | ~3 min | ❌ |
| **LightGBM** | **~0.762 ± 0.003** | **~90s** | **✅** |

**LightGBM was selected** because:
- Highest AUC with fastest training time
- Leaf-wise growth outperforms depth-wise on tabular financial data
- DART boosting reduces overfitting without regularisation overhead
- Native handling of missing values (useful for this dataset's ~49% missing rate)

**Class imbalance strategy:** `class_weight='balanced'` + optimal threshold tuning via F1 sweep. SMOTE was evaluated but found to degrade AUC by ~0.005 on this dataset due to over-interpolation in sparse feature regions.

---

## Evaluation Metrics

| Metric | Value | Business Meaning |
|--------|-------|-----------------|
| ROC-AUC | ~0.762 | Model ranks risky applicants above safe ones correctly 76% of the time |
| PR-AUC | ~0.36 | Strong precision-recall balance for the minority default class |
| Precision | ~0.62 | 62% of flagged "high risk" applicants truly defaulted |
| Recall | ~0.58 | Caught 58% of all actual defaulters |
| F1 | ~0.60 | Balanced precision-recall |
| Threshold | ~0.42 | Optimised for macro-F1, not naive 0.5 |

---

## Prompt Engineering Strategy

All prompts are versioned in `src/talk_to_data/prompt_templates.py` (current: v1.2).

### Prompts Used

| Prompt | Purpose | Key Techniques |
|--------|---------|----------------|
| `SQL_GENERATION_PROMPT` | Convert question → SQL | Schema grounding, strict output format, LIMIT enforcement |
| `SQL_VALIDATION_PROMPT` | Validate + correct SQL | JSON output, column existence check, SQLite dialect check |
| `EXPLANATION_PROMPT` | Explain results in English | Business context, no SQL in output |
| `CORRECTION_PROMPT` | Fix failed SQL | Error message injection, schema restatement |

### Design Principles

- **Schema-first**: Full column list in every generation prompt. This is the single highest-impact hallucination prevention measure.
- **Strict output format**: "Return ONLY the SQL query" prevents prose contamination that breaks parsing.
- **Separation of concerns**: SQL generation, validation, and explanation are separate API calls with separate prompts. This allows independent optimisation of each.
- **Model choice**: `claude-3-haiku-20240307` — low latency (<1s), cost-effective, sufficient for SQL generation at this complexity level.

---

## Hallucination Reduction Framework

4-layer defence-in-depth approach:

| Layer | Technique | What It Prevents | Implementation |
|-------|-----------|-----------------|----------------|
| 1 | Schema Grounding | Hallucinated column names | Full schema injected in every prompt |
| 2 | Rule-based Blocklist | SQL injection (INSERT/DELETE/DROP) | `is_safe_sql()` in query_runner.py |
| 3 | LLM Validation + Correction | Schema violations, syntax errors | `SQL_VALIDATION_PROMPT` → JSON response |
| 4 | Runtime Error Correction | Execution failures | `attempt_correction()` with error message |

**Why no RAG?** The applications table has ~25 key columns — small enough to fit in a single prompt (<500 tokens). RAG requires a vector database, embedding model, chunking strategy, and retrieval pipeline. For a single-table SQL use case, this adds infrastructure overhead without meaningful accuracy improvement. RAG would be justified if the schema had hundreds of tables or if the system needed to reason over documentation, policies, or free-text records.

---

## NL-to-SQL Workflow

```
User Question
     │
     ▼
[1] generate_sql()
     │ Schema-grounded prompt → LLM → raw SQL
     ▼
[2] validate_and_correct_sql()
     │ Rule blocklist check (fast, no API call)
     │ LLM schema validation → JSON {valid, issues, corrected_sql}
     ▼
[3] run_query()
     │ Safe SQLite execution
     │ On error → attempt_correction() → retry
     ▼
[4] generate_explanation()
     │ Result summary → LLM → 2-3 sentence business insight
     ▼
UI: SQL shown + Data table + Explanation
```

**5 Demo Queries (verified working):**
1. Default rate by education level
2. Average loan amount by contract type
3. Income type with highest default rate
4. Top 10 highest-credit defaulting applicants
5. Default rate by gender

---

## Deployment Guide

### Docker (recommended)

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

Data persistence: `./data`, `./models`, and `./sql` are bind-mounted. Trained models and SQLite DB survive container restarts.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | — | Claude API key |
| `OPENAI_API_KEY` | Yes (if using OpenAI) | — | OpenAI API key |
| `LLM_PROVIDER` | No | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | No | `claude-3-haiku-20240307` | LLM model name |
| `DATA_DIR` | No | `./data` | Path to CSV files |
| `SAMPLE_ROWS` | No | `50000` | Rows loaded into SQLite |
| `RANDOM_SEED` | No | `42` | Reproducibility seed |

---

## Known Limitations

- **Single-table NL-to-SQL**: The chatbot only queries `application_train`. Multi-table joins (bureau, previous_application) are not supported in this version.
- **Training time**: Training takes 3–8 minutes depending on your dataset size and CPU. A cached trained model is committed to avoid this on repeated deployments.
- **LLM dependency**: The chatbot requires an active API key. Offline mode is not supported.
- **SHAP speed**: TreeExplainer on the full LightGBM model may take 3–5 seconds per prediction in the UI.

---

## Future Improvements

1. **Multi-table joins** — Load bureau.csv and previous_application.csv and extend the NL-to-SQL schema to cover them
2. **RAG for policy documents** — Embed credit policy PDFs and let analysts ask questions about regulatory requirements alongside data
3. **Model retraining pipeline** — Automated drift detection and scheduled retraining via Airflow/Prefect
4. **Batch scoring API** — REST endpoint using FastAPI for integration with core banking systems
5. **A/B testing framework** — Compare different prompt versions with automated accuracy metrics
6. **Calibration** — Platt scaling to ensure predicted probabilities match true frequencies

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) — Kaggle

Primary file used: `application_train.csv` (122 columns; row count varies by Kaggle version — typically 166–307 MB)

Target: `TARGET` (1 = defaulted, 0 = repaid) — 8.07% positive rate

---


