"""
Shared utility functions used across the project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.utils.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


def get_risk_band(probability: float) -> Dict[str, Any]:
    """
    Convert a raw model probability into a business-readable risk classification.

    Args:
        probability: Float between 0 and 1 (default probability).

    Returns:
        dict with risk_score, risk_band, recommendation, color, and emoji.
    """
    score = round(probability * 100, 1)

    if probability >= HIGH_RISK_THRESHOLD:
        return {
            "risk_score": score,
            "risk_band": "HIGH RISK",
            "recommendation": "Reject application or require significant collateral. "
                              "Immediate manual review by senior credit officer required.",
            "color": "#ef4444",
            "bg_color": "#fef2f2",
            "emoji": "🔴",
            "action": "REJECT / ESCALATE",
        }
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return {
            "risk_score": score,
            "risk_band": "MEDIUM RISK",
            "recommendation": "Manual underwriter review recommended before approval. "
                              "Consider reduced loan amount or additional guarantor.",
            "color": "#f59e0b",
            "bg_color": "#fffbeb",
            "emoji": "🟡",
            "action": "REVIEW",
        }
    else:
        return {
            "risk_score": score,
            "risk_band": "LOW RISK",
            "recommendation": "Eligible for standard automated loan processing. "
                              "Proceed with normal documentation checks.",
            "color": "#22c55e",
            "bg_color": "#f0fdf4",
            "emoji": "🟢",
            "action": "APPROVE",
        }


def days_to_years(days: float) -> float:
    """Convert DAYS_BIRTH or DAYS_EMPLOYED (negative) to positive years."""
    return round(abs(days) / 365.25, 1) if pd.notna(days) else None


def format_currency(amount: float) -> str:
    """Format a number as currency string."""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.0f}"


def summarise_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a compact summary dict for a DataFrame."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include="number").shape[1]),
        "categorical_cols": int(df.select_dtypes(include="object").shape[1]),
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that won't blow up on zero denominators."""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator
