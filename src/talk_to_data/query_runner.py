"""
SQL execution layer — runs queries against SQLite and returns DataFrames.
Handles errors gracefully so the UI always gets a usable response.
"""

import sqlite3
from typing import Tuple

import pandas as pd

from src.utils.config import SQLITE_DB
from src.utils.logger import get_logger

log = get_logger(__name__)

# SQL keywords that are never safe to execute
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
    "ALTER", "TRUNCATE", "REPLACE", "ATTACH",
]


def is_safe_sql(sql: str) -> Tuple[bool, str]:
    """
    Fast rule-based safety check before executing any SQL.

    Returns (is_safe, reason).
    """
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT statements are permitted."

    for keyword in _FORBIDDEN_KEYWORDS:
        if f" {keyword} " in f" {sql_upper} ":
            return False, f"Blocked keyword detected: {keyword}"

    return True, ""


def run_query(sql: str) -> Tuple[pd.DataFrame, str]:
    """
    Execute a SQL query against the SQLite database.

    Args:
        sql: The SQL string to execute.

    Returns:
        (result_df, error_message)
        On success: (DataFrame with results, "")
        On failure: (empty DataFrame, error message)
    """
    # Safety check
    safe, reason = is_safe_sql(sql)
    if not safe:
        log.warning(f"Blocked SQL execution: {reason} | SQL: {sql[:100]}")
        return pd.DataFrame({"message": [f"Query blocked: {reason}"]}), reason

    db_path = str(SQLITE_DB)

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()

        log.info(f"Query executed — {len(df)} rows returned")
        return df, ""

    except Exception as e:
        error_msg = str(e)
        log.error(f"SQL execution error: {error_msg} | SQL: {sql[:200]}")
        return pd.DataFrame(), error_msg


def summarise_result(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Produce a text summary of a query result for injection into LLM prompts.

    Keeps it short to stay within context limits.
    """
    if df.empty:
        return "No results returned."

    lines = [
        f"Rows returned: {len(df)}",
        f"Columns: {', '.join(df.columns.tolist())}",
        "",
        "Sample data (first rows):",
        df.head(max_rows).to_string(index=False),
    ]

    if len(df) > max_rows:
        lines.append(f"... and {len(df) - max_rows} more rows")

    # Numeric summary if applicable
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        stats = df[num_cols].agg(["mean", "min", "max"]).round(3)
        lines.append("\nNumeric summary:")
        lines.append(stats.to_string())

    return "\n".join(lines)
