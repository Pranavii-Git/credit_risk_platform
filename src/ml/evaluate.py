"""
Model evaluation — all metrics and charts needed for the UI and README.
"""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute the full set of evaluation metrics.

    Returns a dict with both raw values and business-readable descriptions.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, probs), 4),
        "pr_auc":    round(average_precision_score(y_test, probs), 4),
        "threshold": round(threshold, 3),
    }

    log.info(f"Evaluation metrics: {metrics}")
    return metrics


def metrics_business_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame that shows metrics with plain-English business meaning."""
    rows = [
        {
            "Metric": "ROC-AUC",
            "Value": f"{metrics['roc_auc']:.4f}",
            "Business Meaning": "Model's ability to rank risky applicants above safe ones (1.0 = perfect)",
        },
        {
            "Metric": "PR-AUC",
            "Value": f"{metrics['pr_auc']:.4f}",
            "Business Meaning": "Precision-Recall balance — more meaningful than AUC when defaults are rare",
        },
        {
            "Metric": "Precision",
            "Value": f"{metrics['precision']:.4f}",
            "Business Meaning": "Of all applicants we flagged as HIGH RISK, this fraction truly defaulted",
        },
        {
            "Metric": "Recall",
            "Value": f"{metrics['recall']:.4f}",
            "Business Meaning": "Of all actual defaulters, this fraction we correctly caught",
        },
        {
            "Metric": "F1 Score",
            "Value": f"{metrics['f1_score']:.4f}",
            "Business Meaning": "Harmonic mean of Precision and Recall — overall balance",
        },
        {
            "Metric": "Accuracy",
            "Value": f"{metrics['accuracy']:.4f}",
            "Business Meaning": "Overall correct predictions (misleading on imbalanced data — use AUC instead)",
        },
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    """ROC curve with AUC annotation."""
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"LightGBM (AUC = {auc:.4f})",
        line=dict(color="#2563eb", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Baseline",
        line=dict(color="#94a3b8", dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.2),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=420,
    )
    return fig


def plot_confusion_matrix(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    labels = ["Repaid (0)", "Defaulted (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix (threshold = {threshold:.2f})",
        height=380,
        paper_bgcolor="#ffffff",
    )
    return fig


def plot_feature_importance(model: Any, feature_names: list, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).nlargest(top_n)

    fig = go.Figure(go.Bar(
        x=fi.values[::-1],
        y=fi.index[::-1],
        orientation="h",
        marker_color="#2563eb",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances (LightGBM)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=500,
    )
    return fig


def plot_pr_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    """Precision-Recall curve."""
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=f"LightGBM (AP = {ap:.4f})",
        line=dict(color="#16a34a", width=2.5),
    ))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=420,
    )
    return fig
