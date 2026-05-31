"""
Exploratory Data Analysis — Credit Risk Dataset
================================================
Converted from eda.ipynb for use in the Streamlit app.

This module provides:
  - run_full_eda(df)  → dict of all charts and insights
  - Individual chart functions callable from UI

All charts are Plotly figures (JSON-serialisable, Streamlit-compatible).
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1d4ed8",
    "danger":  "#dc2626",
    "success": "#16a34a",
    "warning": "#d97706",
    "neutral": "#64748b",
    "bg":      "#ffffff",
}

DEFAULT_COLORSCALE = ["#16a34a", "#d97706", "#dc2626"]

# ── Shared layout defaults applied to every chart ─────────────────────────────
LAYOUT_DEFAULTS = dict(
    plot_bgcolor  = "#ffffff",
    paper_bgcolor = "#ffffff",
    font          = dict(color="#1e293b", size=13, family="DM Sans, sans-serif"),
    title_font    = dict(color="#0f172a", size=15, family="DM Sans, sans-serif"),
    xaxis         = dict(
        tickfont  = dict(color="#1e293b", size=12),
        title_font= dict(color="#1e293b", size=13),
        gridcolor = "#e2e8f0",
        linecolor = "#cbd5e1",
    ),
    yaxis         = dict(
        tickfont  = dict(color="#1e293b", size=12),
        title_font= dict(color="#1e293b", size=13),
        gridcolor = "#e2e8f0",
        linecolor = "#cbd5e1",
    ),
    legend        = dict(
        font      = dict(color="#1e293b", size=12),
        bgcolor   = "#f8fafc",
        bordercolor = "#e2e8f0",
    ),
)


def _apply_defaults(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply shared layout defaults to any figure."""
    fig.update_layout(height=height, **LAYOUT_DEFAULTS)
    fig.update_xaxes(
        tickfont=dict(color="#1e293b", size=12),
        title_font=dict(color="#1e293b", size=13),
        gridcolor="#e2e8f0",
    )
    fig.update_yaxes(
        tickfont=dict(color="#1e293b", size=12),
        title_font=dict(color="#1e293b", size=13),
        gridcolor="#e2e8f0",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Business insights
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_INSIGHTS = [
    {
        "title": "External Credit Scores Are the Strongest Predictors",
        "insight": "EXT_SOURCE_2 and EXT_SOURCE_3 are the top two features by SHAP importance. "
                   "Applicants with average EXT_SOURCE_2 below 0.4 default at nearly 3× the base rate. "
                   "Action: Weight external bureau data heavily in the scoring model.",
        "icon": "📊",
    },
    {
        "title": "8% Overall Default Rate Creates Severe Class Imbalance",
        "insight": "Only 8.07% of applicants defaulted. Training without class balancing causes the model "
                   "to predict 'repaid' for everyone and still achieve 92% accuracy — a misleading metric. "
                   "Action: Use class_weight='balanced' and optimise for ROC-AUC, not accuracy.",
        "icon": "⚖️",
    },
    {
        "title": "DAYS_EMPLOYED Anomaly: 55,374 Records with Sentinel Value",
        "insight": "55,374 rows (18% of data) have DAYS_EMPLOYED = 365,243 — a placeholder for "
                   "'not employed.' This must be replaced with NaN before modelling or it creates "
                   "a spurious 1,000-year employment signal that distorts the model.",
        "icon": "🔍",
    },
    {
        "title": "Younger Applicants Default More",
        "insight": "Applicants aged 20–30 have a default rate of ~11%, nearly double the overall rate. "
                   "This likely reflects less stable income and shorter credit history. "
                   "Action: Age (derived from DAYS_BIRTH) should be a model feature.",
        "icon": "👤",
    },
    {
        "title": "Debt-to-Income Ratio is a Strong Risk Signal",
        "insight": "Applicants with Credit/Income ratio > 3.0 default at 12–15%, well above the 8% baseline. "
                   "This engineered feature (CREDIT_INCOME_RATIO) improves model AUC by ~0.008. "
                   "Action: Include as a feature and set policy thresholds.",
        "icon": "💰",
    },
    {
        "title": "Education Level Inversely Correlates With Default",
        "insight": "Applicants with Higher Education default at ~5.3% vs 10.6% for those with only "
                   "Lower Secondary education — a 2× difference. "
                   "Action: Education level is a significant predictor worth including in scoring.",
        "icon": "🎓",
    },
    {
        "title": "Region Risk Rating Amplifies Default Probability",
        "insight": "Applicants in Region Rating 3 (highest risk) default at 10.4%, vs 5.9% in Rating 1. "
                   "Geographic risk is predictive independent of applicant characteristics.",
        "icon": "🗺️",
    },
    {
        "title": "Property Ownership Reduces Default Risk",
        "insight": "Applicants who own property (FLAG_OWN_REALTY = Y) default at 7.5% vs 8.9% for renters. "
                   "Property ownership signals financial stability and provides collateral.",
        "icon": "🏠",
    },
    {
        "title": "Cash Loans Have Higher Default Rate Than Revolving Loans",
        "insight": "Cash loan applicants default at 8.4% vs 5.4% for revolving credit — suggesting "
                   "revolving credit customers are more financially disciplined or better screened.",
        "icon": "💳",
    },
    {
        "title": "Pensioners Are the Safest Income Segment",
        "insight": "Pensioners default at only ~5.4%, the lowest of any income type. "
                   "Working applicants default at ~8.7%. Business analysts should consider lower "
                   "rates or higher approval limits for pension-income applicants.",
        "icon": "🏦",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Chart functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["TARGET"].value_counts()
    labels = ["Repaid (0)", "Defaulted (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

    fig.add_trace(go.Pie(
        labels=labels, values=values,
        marker_colors=[PALETTE["success"], PALETTE["danger"]],
        hole=0.4,
        textinfo="label+percent",
        textfont=dict(color="#1e293b", size=13),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=[PALETTE["success"], PALETTE["danger"]],
        text=values, textposition="outside",
        textfont=dict(color="#1e293b", size=12),
    ), row=1, col=2)

    fig.update_layout(
        title="Loan Outcome Distribution (Class Imbalance)",
        showlegend=False,
    )
    return _apply_defaults(fig, height=380)


def plot_default_by_education(df: pd.DataFrame) -> go.Figure:
    edu = (
        df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
    )
    edu.columns = ["Education", "Default Rate", "Count"]
    edu["Default Rate %"] = (edu["Default Rate"] * 100).round(2)
    edu = edu.sort_values("Default Rate %", ascending=False)

    fig = px.bar(
        edu, x="Education", y="Default Rate %",
        text="Default Rate %",
        color="Default Rate %",
        color_continuous_scale=DEFAULT_COLORSCALE,
        title="Default Rate by Education Level",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="#1e293b", size=12),
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Education Level",
        yaxis_title="Default Rate (%)",
        coloraxis_colorbar=dict(tickfont=dict(color="#1e293b")),
    )
    return _apply_defaults(fig, height=420)


def plot_default_by_income_type(df: pd.DataFrame) -> go.Figure:
    inc = (
        df.groupby("NAME_INCOME_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
    )
    inc.columns = ["Income Type", "Default Rate", "Count"]
    inc["Default Rate %"] = (inc["Default Rate"] * 100).round(2)
    inc = inc.sort_values("Default Rate %", ascending=False)

    fig = px.bar(
        inc, x="Default Rate %", y="Income Type",
        orientation="h",
        text="Default Rate %",
        color="Default Rate %",
        color_continuous_scale=DEFAULT_COLORSCALE,
        title="Default Rate by Income / Employment Type",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="#1e293b", size=12),
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(autorange="reversed"),
        coloraxis_colorbar=dict(tickfont=dict(color="#1e293b")),
    )
    return _apply_defaults(fig, height=420)


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    df_plot = df.copy()
    df_plot["Age"] = df_plot["DAYS_BIRTH"].abs() / 365.25
    df_plot["Outcome"] = df_plot["TARGET"].map({0: "Repaid", 1: "Defaulted"})

    fig = px.histogram(
        df_plot, x="Age", color="Outcome",
        nbins=40, barmode="overlay", opacity=0.75,
        color_discrete_map={"Repaid": PALETTE["success"], "Defaulted": PALETTE["danger"]},
        title="Age Distribution by Loan Outcome",
    )
    fig.update_layout(xaxis_title="Age (years)")
    return _apply_defaults(fig, height=400)


def plot_credit_income_ratio(df: pd.DataFrame) -> go.Figure:
    df_plot = df.copy()
    df_plot["CREDIT_INCOME_RATIO"] = df_plot["AMT_CREDIT"] / (df_plot["AMT_INCOME_TOTAL"] + 1)
    df_plot = df_plot[df_plot["CREDIT_INCOME_RATIO"] < 20]
    df_plot["Outcome"] = df_plot["TARGET"].map({0: "Repaid", 1: "Defaulted"})

    fig = px.box(
        df_plot, x="Outcome", y="CREDIT_INCOME_RATIO",
        color="Outcome",
        color_discrete_map={"Repaid": PALETTE["success"], "Defaulted": PALETTE["danger"]},
        title="Credit-to-Income Ratio by Loan Outcome",
        points=False,
    )
    fig.update_layout(yaxis_title="Credit / Income Ratio")
    return _apply_defaults(fig, height=400)


def plot_missing_values(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    missing = (
        df.isnull().mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    missing.columns = ["Column", "Missing %"]
    missing["Missing %"] = (missing["Missing %"] * 100).round(2)
    missing = missing[missing["Missing %"] > 0]

    fig = px.bar(
        missing, x="Missing %", y="Column",
        orientation="h", text="Missing %",
        color="Missing %",
        color_continuous_scale=["#fef9c3", "#f59e0b", "#ef4444"],
        title=f"Top {top_n} Columns by Missing Value Rate",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="#1e293b", size=11),
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(autorange="reversed"),
        coloraxis_colorbar=dict(tickfont=dict(color="#1e293b")),
    )
    return _apply_defaults(fig, height=500)


def plot_ext_source_distributions(df: pd.DataFrame) -> go.Figure:
    sources = ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=sources)

    for i, col in enumerate(sources, start=1):
        if col not in df.columns:
            continue
        for outcome, color in [(0, PALETTE["success"]), (1, PALETTE["danger"])]:
            subset = df[df["TARGET"] == outcome][col].dropna()
            label = "Repaid" if outcome == 0 else "Defaulted"
            fig.add_trace(
                go.Violin(
                    y=subset, name=label,
                    box_visible=True,
                    line_color=color,
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )

    fig.update_layout(title="External Credit Score Distributions by Loan Outcome")
    return _apply_defaults(fig, height=430)


def plot_loan_amount_by_outcome(df: pd.DataFrame) -> go.Figure:
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["TARGET"].map({0: "Repaid", 1: "Defaulted"})
    df_plot = df_plot[df_plot["AMT_CREDIT"] < 2_000_000]

    fig = px.histogram(
        df_plot, x="AMT_CREDIT", color="Outcome",
        nbins=50, barmode="overlay", opacity=0.7,
        color_discrete_map={"Repaid": PALETTE["success"], "Defaulted": PALETTE["danger"]},
        title="Loan Amount Distribution by Outcome",
    )
    fig.update_layout(xaxis_title="Loan Amount (USD)")
    return _apply_defaults(fig, height=400)


def plot_days_employed_anomaly(df: pd.DataFrame) -> go.Figure:
    anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
    normal_count  = len(df) - anomaly_count

    fig = go.Figure(go.Bar(
        x=["Normal employment records", "Anomaly (365243 sentinel)"],
        y=[normal_count, anomaly_count],
        marker_color=[PALETTE["success"], PALETTE["danger"]],
        text=[f"{normal_count:,}", f"{anomaly_count:,}"],
        textposition="outside",
        textfont=dict(color="#1e293b", size=13),
    ))
    fig.update_layout(
        title="DAYS_EMPLOYED Anomaly — 365,243 is a Sentinel for 'Not Employed'",
        yaxis_title="Count",
    )
    return _apply_defaults(fig, height=380)


def plot_correlation_with_target(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    num_df = df.select_dtypes(include="number")
    corr = (
        num_df.corr()["TARGET"]
        .drop("TARGET")
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        x=corr.index, y=corr.values,
        labels={"x": "Feature", "y": "|Correlation with TARGET|"},
        color=corr.values,
        color_continuous_scale=DEFAULT_COLORSCALE,
        title=f"Top {top_n} Features Correlated with Loan Default",
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_colorbar=dict(tickfont=dict(color="#1e293b")),
    )
    return _apply_defaults(fig, height=420)


def plot_gender_default(df: pd.DataFrame) -> go.Figure:
    gender = df.groupby("CODE_GENDER")["TARGET"].mean().reset_index()
    gender.columns = ["Gender", "Default Rate"]
    gender["Default Rate %"] = (gender["Default Rate"] * 100).round(2)
    gender["Gender"] = gender["Gender"].map({"M": "Male", "F": "Female"}).fillna(gender["Gender"])

    fig = px.bar(
        gender, x="Gender", y="Default Rate %",
        text="Default Rate %",
        color="Default Rate %",
        color_continuous_scale=DEFAULT_COLORSCALE,
        title="Default Rate by Gender",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="#1e293b", size=13),
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_colorbar=dict(tickfont=dict(color="#1e293b")),
    )
    return _apply_defaults(fig, height=360)


# ─────────────────────────────────────────────────────────────────────────────
# Master EDA runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> Dict[str, Any]:
    log.info("Running EDA ...")

    summary = {
        "total_rows":            len(df),
        "total_columns":         df.shape[1],
        "default_rate":          round(df["TARGET"].mean() * 100, 2),
        "defaulted_count":       int(df["TARGET"].sum()),
        "repaid_count":          int((df["TARGET"] == 0).sum()),
        "missing_cells":         int(df.isnull().sum().sum()),
        "missing_rate":          round(df.isnull().mean().mean() * 100, 2),
        "duplicate_rows":        int(df.duplicated().sum()),
        "numeric_features":      int(df.select_dtypes(include="number").shape[1]),
        "categorical_features":  int(df.select_dtypes(include="object").shape[1]),
        "days_employed_anomaly": int((df["DAYS_EMPLOYED"] == 365243).sum()),
    }

    charts = {
        "target_distribution":   plot_target_distribution(df),
        "default_by_education":  plot_default_by_education(df),
        "default_by_income_type":plot_default_by_income_type(df),
        "age_distribution":      plot_age_distribution(df),
        "credit_income_ratio":   plot_credit_income_ratio(df),
        "missing_values":        plot_missing_values(df),
        "ext_sources":           plot_ext_source_distributions(df),
        "loan_amount":           plot_loan_amount_by_outcome(df),
        "days_employed_anomaly": plot_days_employed_anomaly(df),
        "correlation":           plot_correlation_with_target(df),
        "gender_default":        plot_gender_default(df),
    }

    log.info(f"EDA complete — {len(charts)} charts generated")

    return {
        "summary":  summary,
        "charts":   charts,
        "insights": BUSINESS_INSIGHTS,
    }