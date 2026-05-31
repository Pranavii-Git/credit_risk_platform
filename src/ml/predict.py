"""
Inference pipeline — loads trained model bundle and scores new applicants.
Also wraps SHAP explanations for the UI.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.config import MODEL_PATH, TARGET_COL, ID_COL
from src.utils.helpers import get_risk_band
from src.utils.logger import get_logger

log = get_logger(__name__)

# Module-level cache — loaded once per Streamlit session
_bundle_cache: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_bundle() -> Dict[str, Any]:
    """Load and cache the saved model bundle."""
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache

    path = Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run the training pipeline first: python -m src.ml.train"
        )

    log.info(f"Loading model bundle from {path} ...")
    _bundle_cache = joblib.load(path)
    log.info("Model bundle loaded successfully.")
    return _bundle_cache


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Predict risk for a single applicant.

    Args:
        features: Single-row DataFrame (already preprocessed).

    Returns:
        dict with probability, risk_band, recommendation, and SHAP explanation.
    """
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]

    prob = model.predict_proba(features)[0, 1]
    risk = get_risk_band(prob)
    shap_explanation = get_shap_explanation(model, features, bundle["feature_names"])

    return {
        "probability": float(prob),
        "threshold": threshold,
        **risk,
        "shap_top_features": shap_explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanations
# ─────────────────────────────────────────────────────────────────────────────

def get_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)

        # Align columns to training order
        X_aligned = X.reindex(columns=feature_names, fill_value=0)
        shap_values = explainer.shap_values(X_aligned)

        # For binary classifiers, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        results = []
        for feat, val in zip(feature_names, sv):
            results.append({
                "feature": feat,
                "shap_value": float(val),
                "direction": "increases risk" if val > 0 else "decreases risk",
                "abs_value": abs(float(val)),
                "business_label": _feature_business_label(feat),
            })

        return sorted(results, key=lambda x: x["abs_value"], reverse=True)[:top_n]

    except Exception as e:
        log.warning(f"SHAP explanation failed: {e}")
        return []


def _feature_business_label(feature_name: str) -> str:
    """Map technical feature names to human-readable labels."""
    mapping = {
        "EXT_SOURCE_1": "External Credit Score 1",
        "EXT_SOURCE_2": "External Credit Score 2",
        "EXT_SOURCE_3": "External Credit Score 3",
        "CREDIT_INCOME_RATIO": "Debt-to-Income Ratio",
        "ANNUITY_INCOME_RATIO": "Monthly Payment Burden",
        "AMT_CREDIT": "Loan Amount",
        "AMT_INCOME_TOTAL": "Annual Income",
        "DAYS_BIRTH": "Applicant Age",
        "AGE_YEARS": "Applicant Age (Years)",
        "DAYS_EMPLOYED": "Employment Duration",
        "DAYS_EMPLOYED_RATIO": "Employment Stability Ratio",
        "AMT_ANNUITY": "Monthly Annuity",
        "CNT_CHILDREN": "Number of Children",
        "CREDIT_TERM": "Implied Loan Term",
        "INCOME_PER_FAMILY": "Income Per Family Member",
        "REGION_RATING_CLIENT": "Region Risk Rating",
    }
    return mapping.get(feature_name, feature_name.replace("_", " ").title())


def plot_shap_waterfall(shap_features: List[Dict[str, Any]], base_value: float = 0.5) -> go.Figure:
    """
    Plotly waterfall chart showing each feature's contribution to the risk score.
    """
    if not shap_features:
        return go.Figure()

    top = shap_features[:8]
    labels = [f["business_label"] for f in top]
    values = [f["shap_value"] for f in top]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Contributions (Red = Increases Risk | Green = Decreases Risk)",
        xaxis_title="SHAP Value (impact on default probability)",
        yaxis_title="",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    return fig
