"""
Credit Risk Intelligence Platform — Streamlit Application
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background-color: #dde3f0 !important; }
    .stApp > header { background-color: #dde3f0 !important; }
    .main { background-color: #dde3f0; }
    [data-testid="stAppViewContainer"] { background-color: #dde3f0 !important; }
    [data-testid="stHeader"] { background-color: #dde3f0 !important; }
    [data-testid="block-container"] {
        background-color: #dde3f0 !important;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] > div > div {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #166534 !important;
        border: 1px solid #22c55e !important;
    }
    section[data-testid="stSidebar"] [data-testid="stAlertWarning"] {
        background-color: #854d0e !important;
        border: 1px solid #f59e0b !important;
    }
    section[data-testid="stSidebar"] [data-testid="stAlertError"] {
        background-color: #7f1d1d !important;
        border: 1px solid #ef4444 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] hr { border-color: #334155 !important; }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background-color: #1d4ed8 !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(37,99,235,0.35);
    }
    .hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff; }
    .hero-banner p  { font-size: 1rem; opacity: 0.9; margin: 0.4rem 0 0; color: #ffffff; }

    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2563eb;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 0.75rem;
        border: 1px solid #cbd5e1;
    }
    .metric-card.danger  { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #22c55e; }
    .metric-card.warning { border-left-color: #f59e0b; }
    .metric-card h3 {
        font-size: 0.78rem; color: #475569; margin: 0;
        text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
    .metric-card .sub   { font-size: 0.82rem; color: #64748b; }

    .insight-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        border: 1px solid #cbd5e1;
    }
    .insight-card h4 { margin: 0 0 0.4rem; color: #0f172a; font-size: 0.95rem; font-weight: 600; }
    .insight-card p  { margin: 0; color: #334155; font-size: 0.88rem; line-height: 1.5; }

    .chat-user {
        background: #2563eb; color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        max-width: 80%; margin-left: auto; font-size: 0.9rem;
    }
    .chat-bot {
        background: white; border: 1px solid #cbd5e1;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
        max-width: 90%; font-size: 0.9rem; color: #1e293b;
    }
    .sql-block {
        background: #1e293b; color: #7dd3fc;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem; margin: 0.5rem 0; overflow-x: auto;
    }

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #0f172a;
        margin-bottom: 0.75rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid #94a3b8;
    }

    [data-testid="stPlotlyChart"] {
        background: #ffffff !important; border-radius: 12px !important;
        padding: 1rem !important; box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
        border: 1px solid #cbd5e1 !important; margin-bottom: 1rem !important;
    }

    .stButton > button {
        background-color: #2563eb !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; font-size: 0.9rem !important;
    }
    .stButton > button:hover { background-color: #1d4ed8 !important; }

    .stTextInput > div > div > input {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    }
    .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
    .stNumberInput > div > div > input {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stSelectbox > div > div {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] { background-color: #dde3f0 !important; }
    [data-baseweb="tab"] { color: #1e293b !important; font-weight: 500 !important; }

    [data-testid="stDataFrame"] {
        background: #ffffff !important; border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    }

    [data-testid="stAlert"] {
        background-color: #eff6ff !important; color: #1e293b !important;
        border: 1px solid #bfdbfe !important; border-radius: 8px !important;
    }

    p, span, label { color: #1e293b; }
    h1, h2, h3, h4, h5 { color: #0f172a; font-weight: 700; }

    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

from src.utils.config import DATA_DIR, MODEL_PATH, TRAIN_FILE
from src.utils.logger import get_logger
log = get_logger("app")


@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    from src.data.loader import load_train
    df = load_train()
    sample_n = min(80_000, len(df))
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    return df

@st.cache_data(show_spinner="Running EDA …")
def get_eda(_df: pd.DataFrame) -> dict:
    from notebooks.eda import run_full_eda
    return run_full_eda(_df)

@st.cache_resource(show_spinner="Loading model …")
def get_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)

def model_is_trained() -> bool:
    return Path(MODEL_PATH).exists()

def dataset_exists() -> bool:
    return (Path(DATA_DIR) / TRAIN_FILE).exists()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#ffffff; font-weight:800; font-size:1.3rem; margin:0;'>🏦 Credit Risk Platform</h2>", unsafe_allow_html=True)
    st.markdown("---")

    if dataset_exists():
        st.success("✅ Dataset loaded")
    else:
        st.error("❌ Dataset not found\nPlace `application_train.csv` in `./data/`")

    if model_is_trained():
        st.success("✅ Model trained")
    else:
        st.warning("⚠️ Model not trained yet")

    st.markdown("---")

    if dataset_exists() and not model_is_trained():
        if st.button("🚀 Train Model Now", use_container_width=True, type="primary"):
            with st.spinner("Training model — this takes 2–5 minutes …"):
                try:
                    from src.data.loader import load_train, load_or_create_sqlite
                    from src.data.preprocessor import preprocess
                    from src.ml.train import run_training
                    df_raw = load_train()
                    df_proc, artifacts = preprocess(df_raw)
                    run_training(df_proc, artifacts)
                    load_or_create_sqlite()
                    st.success("✅ Training complete!")
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    elif dataset_exists() and model_is_trained():
        if st.button("🔄 Retrain Model", use_container_width=True):
            with st.spinner("Retraining …"):
                try:
                    from src.data.loader import load_train, load_or_create_sqlite
                    from src.data.preprocessor import preprocess
                    from src.ml.train import run_training
                    df_raw = load_train()
                    df_proc, artifacts = preprocess(df_raw)
                    run_training(df_proc, artifacts)
                    load_or_create_sqlite(force_refresh=True)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Retrain complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retrain failed: {e}")

    st.markdown("---")
    st.caption("**Stack:** LightGBM · SHAP · Groq AI · SQLite · Streamlit · Docker")
    st.caption("**Dataset:** Home Credit Default Risk (Kaggle)")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🏦 Credit Risk Intelligence Platform</h1>
  <p>AI-powered loan default prediction · Explainable AI · Natural Language Analytics</p>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Dashboard",
    "🎯 Risk Prediction",
    "💬 Talk to Data",
    "💡 Business Insights",
    "📋 Project Overview",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not dataset_exists():
        st.info("EDA Dashboard requires the dataset locally. "
                "See README for setup instructions.")
        st.stop()

    df  = load_data()
    eda = get_eda(df)
    s   = eda["summary"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>Total Applicants</h3><div class="value">{s['total_rows']:,}</div>
            <div class="sub">sample loaded</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card danger">
            <h3>Default Rate</h3><div class="value">{s['default_rate']}%</div>
            <div class="sub">{s['defaulted_count']:,} defaulted</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h3>Features</h3><div class="value">{s['total_columns']}</div>
            <div class="sub">{s['numeric_features']} numeric · {s['categorical_features']} cat</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card warning">
            <h3>Missing Rate</h3><div class="value">{s['missing_rate']}%</div>
            <div class="sub">{s['missing_cells']:,} cells</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card danger">
            <h3>Anomaly Rows</h3><div class="value">{s['days_employed_anomaly']:,}</div>
            <div class="sub">DAYS_EMPLOYED=365243</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(eda["charts"]["target_distribution"], use_container_width=True)
    with col2: st.plotly_chart(eda["charts"]["default_by_education"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(eda["charts"]["default_by_income_type"], use_container_width=True)
    with col2: st.plotly_chart(eda["charts"]["age_distribution"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(eda["charts"]["credit_income_ratio"], use_container_width=True)
    with col2: st.plotly_chart(eda["charts"]["days_employed_anomaly"], use_container_width=True)

    st.plotly_chart(eda["charts"]["ext_sources"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(eda["charts"]["missing_values"], use_container_width=True)
    with col2: st.plotly_chart(eda["charts"]["correlation"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(eda["charts"]["loan_amount"], use_container_width=True)
    with col2: st.plotly_chart(eda["charts"]["gender_default"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTION + MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    import os
    st.write("CWD:", os.getcwd())
    st.write("Model path from config:", str(MODEL_PATH))
    st.write("Model exists:", Path(MODEL_PATH).exists())
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".joblib"):
                st.write("Model found at:", os.path.join(root, file))

    bundle        = get_model_bundle()
    model         = bundle["model"]
    threshold     = bundle["threshold"]
    feature_names = bundle["feature_names"]
    artifacts     = bundle["preprocessing_artifacts"]
    test_metrics  = bundle.get("test_metrics", {})
    cv_results    = bundle.get("cv_results", None)

    # ── Selected Model Banner ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1e40af; border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;
                box-shadow:0 4px 16px rgba(37,99,235,0.3);">
        <div style="color:#ffffff; font-size:1rem; font-weight:700; margin-bottom:0.3rem;">
            ✅ Selected Model: LightGBM (DART Boosting)
        </div>
        <div style="color:#bfdbfe; font-size:0.85rem;">
            Chosen for best ROC-AUC on 307K applicant records with class_weight=balanced for imbalance handling
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Accuracy Metric Cards ─────────────────────────────────────────────────
    if test_metrics:
        st.markdown('<div class="section-title">Model Accuracy — Held-Out Test Set (20%)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""<div class="metric-card success">
                <h3>ROC-AUC</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('roc_auc','N/A')}</div>
                <div class="sub">Primary metric</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <h3>PR-AUC</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('pr_auc','N/A')}</div>
                <div class="sub">Imbalance-aware</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <h3>F1 Score</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('f1_score','N/A')}</div>
                <div class="sub">Precision+Recall</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <h3>Precision</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('precision','N/A')}</div>
                <div class="sub">Of flagged risky</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card">
                <h3>Recall</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('recall','N/A')}</div>
                <div class="sub">Defaults caught</div></div>""", unsafe_allow_html=True)
        with m6:
            st.markdown(f"""<div class="metric-card warning">
                <h3>Threshold</h3>
                <div class="value" style="font-size:1.6rem;">{test_metrics.get('threshold','N/A')}</div>
                <div class="sub">Optimised F1</div></div>""", unsafe_allow_html=True)

        st.info(
            f"**Accuracy = {test_metrics.get('accuracy','N/A')}** — misleading here because 92% of applicants repaid. "
            "Always use ROC-AUC as the primary evaluation metric for imbalanced datasets."
        )
    else:
        st.info("Retrain the model to see live accuracy metrics from your dataset.")

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Model Comparison — All 3 Models Evaluated</div>', unsafe_allow_html=True)

    if cv_results is not None and not cv_results.empty:
        display_cv = cv_results.copy()
        display_cv["Selected"] = display_cv["Model"].apply(
            lambda x: "✅ SELECTED" if "LightGBM" in x else ""
        )
        st.dataframe(display_cv, use_container_width=True, hide_index=True)
    else:
        st.dataframe(pd.DataFrame({
            "Model":               ["LightGBM ✅", "XGBoost", "Logistic Regression"],
            "CV ROC-AUC":          ["0.7580 ± 0.0017", "0.7606 ± 0.0007", "0.5922 ± 0.0058"],
            "Training Time":       ["~3 min", "~1 min", "~11 min (slow)"],
            "Class Imbalance":     ["class_weight=balanced", "scale_pos_weight=11", "class_weight=balanced"],
            "Missing Values":      ["Native handling ✅", "Requires imputation", "Requires imputation"],
            "Interpretability":    ["SHAP (tree)", "SHAP (tree)", "Coefficients (linear)"],
            "Business Suitability":["✅ Best overall", "✅ Good", "❌ Too simple"],
            "Selected":            ["✅ YES", "❌ No", "❌ No"],
        }), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:10px;
                padding:1rem 1.5rem; margin:0.5rem 0;">
        <div style="color:#166534; font-weight:700; margin-bottom:0.4rem;">
            🏆 Why LightGBM was Selected
        </div>
        <div style="color:#14532d; font-size:0.88rem; line-height:1.6;">
            • <b>Best balance</b> of AUC and training speed<br>
            • <b>Leaf-wise growth</b> outperforms XGBoost on tabular financial data<br>
            • <b>DART boosting</b> reduces overfitting without extra tuning<br>
            • <b>Native missing value handling</b> — critical (24% missing in this dataset)<br>
            • <b>SHAP compatible</b> — exact TreeExplainer explanations per prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction Form ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Applicant Risk Assessment</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Financial**")
            amt_credit  = st.number_input("Loan Amount ($)", 10000, 4000000, 500000, step=10000)
            amt_income  = st.number_input("Annual Income ($)", 10000, 10000000, 150000, step=5000)
            amt_annuity = st.number_input("Monthly Annuity ($)", 1000, 200000, 20000, step=1000)
            amt_goods   = st.number_input("Goods Price ($)", 10000, 4000000, 400000, step=10000)

        with c2:
            st.markdown("**Demographics**")
            age_years     = st.slider("Age (years)", 18, 75, 35)
            days_birth    = -int(age_years * 365.25)
            gender        = st.selectbox("Gender", ["F", "M"])
            education     = st.selectbox("Education", [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary", "Academic degree",
            ])
            family_status = st.selectbox("Family Status", [
                "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
            ])
            cnt_children  = st.number_input("Children", 0, 10, 0)

        with c3:
            st.markdown("**Employment & Credit**")
            income_type      = st.selectbox("Income Type", [
                "Working", "Commercial associate", "Pensioner",
                "State servant", "Unemployed", "Student",
            ])
            employment_years = st.slider("Employment Duration (years)", 0, 40, 5)
            days_employed    = -int(employment_years * 365.25) if employment_years > 0 else 365243
            own_car          = st.selectbox("Owns Car", ["N", "Y"])
            own_realty       = st.selectbox("Owns Property", ["N", "Y"])
            region_rating    = st.selectbox("Region Risk Rating", [1, 2, 3])
            ext_source_2     = st.slider("External Credit Score 2", 0.0, 1.0, 0.55, 0.01)
            ext_source_3     = st.slider("External Credit Score 3", 0.0, 1.0, 0.50, 0.01)

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True, type="primary")

    if submitted:
        from src.data.preprocessor import preprocess_single_record
        from src.utils.helpers import get_risk_band

        record = {
            "AMT_CREDIT": amt_credit, "AMT_INCOME_TOTAL": amt_income,
            "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods,
            "CODE_GENDER": gender, "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty, "CNT_CHILDREN": cnt_children,
            "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": family_status,
            "NAME_INCOME_TYPE": income_type, "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed, "REGION_RATING_CLIENT": region_rating,
            "EXT_SOURCE_2": ext_source_2, "EXT_SOURCE_3": ext_source_3,
            "EXT_SOURCE_1": 0.5, "CNT_FAM_MEMBERS": cnt_children + 1.0,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_HOUSING_TYPE": "House / apartment",
            "OCCUPATION_TYPE": "Laborers",
            "REGION_RATING_CLIENT_W_CITY": region_rating,
        }

        try:
            feat_df = preprocess_single_record(record, artifacts)
            for col in feature_names:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df[feature_names]

            prob = float(model.predict_proba(feat_df)[0, 1])
            risk = get_risk_band(prob)
            score_color = risk["color"]

            st.markdown("---")
            st.markdown('<div class="section-title">Risk Assessment Result</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns([2, 2, 3])
            with r1:
                st.markdown(f"""<div class="metric-card" style="border-left-color:{score_color}; text-align:center;">
                    <h3>Risk Score</h3>
                    <div class="value" style="color:{score_color}; font-size:3rem;">{risk['risk_score']}%</div>
                    <div class="sub">Default Probability</div></div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card" style="border-left-color:{score_color}; text-align:center;">
                    <h3>Risk Band</h3>
                    <div class="value" style="color:{score_color};">{risk['emoji']} {risk['risk_band']}</div>
                    <div class="sub">Decision: {risk['action']}</div></div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card" style="border-left-color:{score_color};">
                    <h3>Recommendation</h3>
                    <div style="color:#1e293b; font-size:0.95rem; margin-top:0.3rem; line-height:1.5;">
                        {risk['recommendation']}</div></div>""", unsafe_allow_html=True)

            # SHAP — safe import
            st.markdown('<div class="section-title" style="margin-top:1.5rem;">Explainable AI — Feature Contributions</div>', unsafe_allow_html=True)
            try:
                from src.ml.predict import get_shap_explanation, plot_shap_waterfall
                shap_feats = get_shap_explanation(model, feat_df, feature_names)
                if shap_feats:
                    st.plotly_chart(plot_shap_waterfall(shap_feats), use_container_width=True)
                    st.markdown("**Top Contributing Factors:**")
                    for item in shap_feats[:5]:
                        icon = "🔴" if item["shap_value"] > 0 else "🟢"
                        st.markdown(
                            f"{icon} **{item['business_label']}** — "
                            f"{item['direction']} (SHAP: {item['shap_value']:+.4f})"
                        )
            except Exception as shap_err:
                st.info(
                    "SHAP explanation unavailable due to a system library conflict (PyTorch DLL). "
                    "Fix: run `pip uninstall torch -y` then restart. Risk score above is still accurate."
                )
                log.warning(f"SHAP failed: {shap_err}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            log.exception("Prediction failed")

    # ── Evaluation Charts ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Model Evaluation Charts</div>', unsafe_allow_html=True)

    if st.button("📊 Show Confusion Matrix & ROC Curve"):
        from src.data.preprocessor import preprocess
        from src.ml.evaluate import (
            compute_metrics, metrics_business_table,
            plot_roc_curve, plot_confusion_matrix,
            plot_feature_importance, plot_pr_curve,
        )
        from src.ml.train import split_features_target
        from sklearn.model_selection import train_test_split
        from src.utils.config import TEST_SIZE, RANDOM_SEED

        with st.spinner("Computing on held-out test set …"):
            df_raw  = load_data()
            df_proc, arts = preprocess(df_raw)
            X, y    = split_features_target(df_proc)
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
            )
            for col in feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test    = X_test[feature_names]
            metrics   = compute_metrics(model, X_test, y_test, threshold)
            biz_table = metrics_business_table(metrics)

        st.dataframe(biz_table, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(plot_roc_curve(model, X_test, y_test), use_container_width=True)
        with col2: st.plotly_chart(plot_pr_curve(model, X_test, y_test), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(plot_confusion_matrix(model, X_test, y_test, threshold), use_container_width=True)
        with col2: st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NL-TO-SQL CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    from src.talk_to_data.prompt_templates import DEMO_QUERIES
    from src.utils.config import SQLITE_DB, ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, LLM_PROVIDER

    st.markdown('<div class="section-title">Talk to Your Data</div>', unsafe_allow_html=True)
    st.caption("Ask questions in plain English. The system converts them to SQL and explains the results.")

    if not Path(SQLITE_DB).exists():
        st.warning("SQLite database not yet created. Train the model first, or click below.")
        if st.button("Create Database Now"):
            from src.data.loader import load_or_create_sqlite
            with st.spinner("Loading data into SQLite …"):
                load_or_create_sqlite()
            st.success("Database created!")
            st.rerun()

    api_ok = (
        (LLM_PROVIDER == "groq"      and GROQ_API_KEY) or
        (LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY) or
        (LLM_PROVIDER == "openai"    and OPENAI_API_KEY)
    )

    if not api_ok:
        st.error(
            f"LLM API key not configured for provider '{LLM_PROVIDER}'.\n"
            "Add `GROQ_API_KEY=gsk_...` to your `.env` file and restart."
        )

    # ── Initialise session state ──────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "nl_question" not in st.session_state:
        st.session_state.nl_question = ""

    # ── Demo buttons ──────────────────────────────────────────────────────────
    st.markdown("**💡 Try a demo query:**")
    cols = st.columns(4)
    for i, demo in enumerate(DEMO_QUERIES[:8]):
        if cols[i % 4].button(demo["label"], key=f"demo_{i}"):
            st.session_state.nl_question = demo["question"]

    # ── Input ─────────────────────────────────────────────────────────────────
    question = st.text_input(
        "Your question:",
        value=st.session_state.nl_question,
        placeholder="e.g. What is the default rate by education level?",
        key="nl_input",
    )

    ask_col, clear_col = st.columns([5, 1])
    with ask_col:
        ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary")
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.nl_question  = ""

    # ── Process question ──────────────────────────────────────────────────────
    if ask_btn and question.strip() and api_ok:
        from src.talk_to_data.nl_to_sql import process_question
        with st.spinner("Thinking …"):
            result = process_question(question.strip())
        st.session_state.chat_history.append(result)
        st.session_state.nl_question = ""

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        for entry in st.session_state.chat_history:
            # User bubble
            st.markdown(
                f'<div class="chat-user">❓ {entry["question"]}</div>',
                unsafe_allow_html=True,
            )

            if entry.get("error") and entry["results"] is None:
                st.error(f"❌ {entry['explanation']}")
            else:
                # SQL
                with st.expander("🔍 View Generated SQL", expanded=False):
                    st.markdown(
                        f'<div class="sql-block">{entry["sql"]}</div>',
                        unsafe_allow_html=True,
                    )

                # Data table
                if entry["results"] is not None and not entry["results"].empty:
                    st.dataframe(entry["results"].head(50), use_container_width=True)
                    st.caption(f"{entry['row_count']} rows returned")
                else:
                    st.warning("Query returned no results.")

                # Explanation bubble
                if entry["explanation"]:
                    st.markdown(
                        f'<div class="chat-bot">💡 {entry["explanation"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    from notebooks.eda import BUSINESS_INSIGHTS

    st.markdown('<div class="section-title">Top Business Insights</div>', unsafe_allow_html=True)
    st.caption("Insights derived from EDA and model analysis — actionable for credit policy teams.")

    for i, insight in enumerate(BUSINESS_INSIGHTS, 1):
        st.markdown(f"""<div class="insight-card">
            <h4>{insight['icon']} {i}. {insight['title']}</h4>
            <p>{insight['insight']}</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Business Decision Rules (Derived from ML)</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Rule": ["Automatic Reject", "Manual Review Required", "Conditional Approve", "Automatic Approve"],
        "Condition": [
            "Risk Score ≥ 60% OR (EXT_SOURCE_2 < 0.2 AND AMT_CREDIT > $500K)",
            "Risk Score 35–60% OR Region Rating = 3 with Score > 25%",
            "Risk Score 20–35% with Stable Employment > 2 years",
            "Risk Score < 20% AND EXT_SOURCE_2 > 0.6",
        ],
        "Action": [
            "Decline loan. Inform applicant with reason code.",
            "Assign to senior underwriter. Decision within 48h.",
            "Approve with standard documentation check.",
            "Fast-track approval. Same-day processing.",
        ],
        "Expected Default Rate": ["~35–70%", "~12–35%", "~5–12%", "~1–5%"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Model Comparison Summary</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Model":            ["LightGBM ✅", "XGBoost", "Logistic Regression"],
        "CV ROC-AUC":       ["0.7580 ± 0.0017", "0.7606 ± 0.0007", "0.5922 ± 0.0058"],
        "Training Time":    ["~3 min", "~1 min", "~11 min"],
        "Interpretability": ["SHAP", "SHAP", "Coefficients"],
        "Selected":         ["✅ YES", "❌ No", "❌ No"],
    }), use_container_width=True, hide_index=True)

    st.info(
        "**Why LightGBM?** Best balance of AUC, training speed, and native missing-value handling. "
        "DART boosting prevents overfitting. Fully SHAP-compatible for explainability."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Architecture Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px;
                padding:1.5rem 2rem; box-shadow:0 4px 16px rgba(0,0,0,0.08);">
        <table style="width:100%; border-collapse:separate; border-spacing:10px;
                      font-family:'DM Sans',sans-serif; font-size:0.88rem;">
            <tr>
                <td colspan="3" style="text-align:center; background:#1e40af; color:white;
                    padding:0.8rem; border-radius:8px; font-weight:700; font-size:1rem;">
                    🏦 Credit Risk Intelligence Platform
                </td>
            </tr>
            <tr>
                <td style="background:#dbeafe; border:2px solid #93c5fd; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    📄 CSV Data<br><span style="font-size:0.75rem;color:#64748b;">307K applicant rows</span>
                </td>
                <td style="background:#dcfce7; border:2px solid #86efac; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    🗄️ SQLite DB<br><span style="font-size:0.75rem;color:#64748b;">50K rows for chatbot</span>
                </td>
                <td style="background:#fef9c3; border:2px solid #fde047; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    🤖 LLM (Groq Free)<br><span style="font-size:0.75rem;color:#64748b;">Llama 3.1 8B Instant</span>
                </td>
            </tr>
            <tr>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
            </tr>
            <tr>
                <td style="background:#f1f5f9; border:2px solid #cbd5e1; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    loader.py<br>preprocessor.py
                </td>
                <td style="background:#f1f5f9; border:2px solid #cbd5e1; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    query_runner.py
                </td>
                <td style="background:#f1f5f9; border:2px solid #cbd5e1; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    nl_to_sql.py<br><span style="font-size:0.75rem;color:#64748b;">4-Layer Pipeline</span>
                </td>
            </tr>
            <tr>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
                <td style="text-align:center;color:#94a3b8;font-size:1.4rem;padding:0.2rem;">↓</td>
            </tr>
            <tr>
                <td style="background:#f1f5f9; border:2px solid #cbd5e1; border-radius:8px;
                    padding:0.8rem; text-align:center; color:#1e293b; font-weight:600;">
                    train.py · predict.py<br>evaluate.py · SHAP
                </td>
                <td colspan="2" style="background:#1e40af; border:2px solid #3b82f6; border-radius:8px;
                    padding:0.8rem; text-align:center; color:white; font-weight:600;">
                    🖥️ Streamlit UI (app.py)<br>
                    <span style="font-size:0.8rem;font-weight:400;">
                        EDA · Risk Prediction · Talk to Data · Insights · Overview
                    </span>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📦 Tech Stack**")
        st.dataframe(pd.DataFrame({
            "Component":  ["ML Model", "Explainable AI", "NL-to-SQL LLM", "Database", "UI", "Deployment"],
            "Choice":     ["LightGBM", "SHAP TreeExplainer", "Groq Llama 3.1 (FREE)", "SQLite", "Streamlit", "Docker + Compose"],
            "Reason":     [
                "Best AUC on tabular credit data",
                "Exact tree-based explanations",
                "Free, <1s latency, no credit card",
                "Zero-setup, portable",
                "Rapid ML dashboard",
                "Single-command deploy",
            ],
        }), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**🛡️ Hallucination Reduction — 4 Layers**")
        st.dataframe(pd.DataFrame({
            "Layer":    ["1. Schema Grounding", "2. Rule Blocklist", "3. LLM Validation", "4. Self-Correction"],
            "Prevents": ["Hallucinated columns", "SQL injection", "Schema violations", "Execution errors"],
        }), use_container_width=True, hide_index=True)
        st.info("**No RAG needed** — schema fits in one prompt (<500 tokens).")

    st.markdown("**📁 Project Structure**")
    st.code("""\
credit_risk_platform/
├── data/                      ← Place CSV files here (not committed)
├── documents/                 ← Project presentation PDF
├── notebooks/
│   └── eda.py                 ← EDA charts and insights
├── src/
│   ├── data/
│   │   ├── loader.py          ← CSV + SQLite loading
│   │   └── preprocessor.py    ← Full preprocessing pipeline
│   ├── ml/
│   │   ├── train.py           ← 3-model training + CV
│   │   ├── predict.py         ← Inference + SHAP
│   │   └── evaluate.py        ← Metrics + charts
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py       ← NL-to-SQL pipeline
│   │   ├── query_runner.py    ← Safe SQL execution
│   │   └── prompt_templates.py← Versioned prompts
│   └── utils/
│       ├── config.py          ← All env vars
│       ├── logger.py          ← Logging
│       └── helpers.py         ← Risk scoring
├── app.py                     ← Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md""", language="text")

    st.markdown("---")
    st.markdown("**🚀 Quick Start**")
    st.markdown("""
    <div style="background:#1e293b; border-radius:12px; padding:1.2rem 2rem;
                font-family:'DM Mono',monospace; font-size:0.82rem; color:#86efac;
                line-height:1.8;">
    <pre style="margin:0; color:#86efac; background:transparent;">
# 1. Clone and configure
git clone https://github.com/Pranavii-Git/credit_risk_platform.git
cp .env.example .env  # add GROQ_API_KEY

# 2. Add dataset → ./data/application_train.csv

# 3. Run
docker-compose up --build
# OR: streamlit run app.py → http://localhost:8501
    </pre></div>
    """, unsafe_allow_html=True)
