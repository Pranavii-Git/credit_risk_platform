"""
NL-to-SQL pipeline orchestrator.

Flow:
  User Question
    → Safety pre-check
    → LLM SQL Generation (schema-grounded prompt)
    → Rule-based SQL validation (blocklist)
    → LLM SQL validation + correction (schema check)
    → SQL Execution
    → Business Explanation (separate LLM call)
    → Return structured result

Hallucination reduction layers:
  1. Schema injected into every prompt (grounding)
  2. Rule-based blocklist (no INSERT/DELETE/DROP)
  3. LLM validation prompt with correction
  4. Runtime error caught → correction attempt
  5. Graceful fallback message shown to user
"""

import json
from typing import Dict, Any, Optional

import anthropic

from src.talk_to_data.prompt_templates import (
    SCHEMA_DESCRIPTION,
    SQL_GENERATION_PROMPT,
    SQL_VALIDATION_PROMPT,
    EXPLANATION_PROMPT,
    CORRECTION_PROMPT,
)
from src.talk_to_data.query_runner import run_query, summarise_result, is_safe_sql
from src.utils.config import (
    LLM_PROVIDER, LLM_MODEL,
    GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_client():
    """Return the configured LLM client."""
    if LLM_PROVIDER == "groq":
        from groq import Groq
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        return Groq(api_key=GROQ_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif LLM_PROVIDER == "openai":
        import openai
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")
        return openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    """Unified LLM call — works with Groq, Anthropic, or OpenAI."""
    client = _get_llm_client()

    if LLM_PROVIDER == "groq":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # low temp = more deterministic SQL
        )
        return response.choices[0].message.content.strip()

    elif LLM_PROVIDER == "anthropic":
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    elif LLM_PROVIDER == "openai":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(question: str) -> str:
    """Step 1: Generate SQL from natural language question."""
    prompt = SQL_GENERATION_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        question=question,
    )
    sql = _call_llm(prompt, max_tokens=300)

    # Strip markdown code fences if LLM adds them despite instructions
    sql = sql.replace("```sql", "").replace("```", "").strip()
    log.info(f"Generated SQL: {sql[:120]}")
    return sql


def validate_and_correct_sql(sql: str, question: str) -> str:
    """
    Step 2: Validate SQL with LLM and correct if needed.

    Returns corrected SQL (or original if valid).
    """
    # Fast rule-based check first (no API call)
    safe, reason = is_safe_sql(sql)
    if not safe:
        log.warning(f"Rule-based block: {reason}")
        return f"SELECT 'Blocked: {reason}' AS message;"

    # LLM-based schema validation
    prompt = SQL_VALIDATION_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        sql=sql,
    )

    try:
        response = _call_llm(prompt, max_tokens=400)
        # Strip markdown fences from JSON response
        response = response.replace("```json", "").replace("```", "").strip()
        validation = json.loads(response)

        if not validation.get("valid", True):
            corrected = validation.get("corrected_sql", sql)
            issues = validation.get("issues", [])
            log.info(f"SQL corrected. Issues: {issues}")
            return corrected.strip()

    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Validation parse failed ({e}) — using original SQL")

    return sql


def generate_explanation(question: str, df, sql: str) -> str:
    """Step 3: Generate a business explanation of the results."""
    if df is None or df.empty:
        return "The query returned no results. Try rephrasing your question."

    summary = summarise_result(df)
    prompt = EXPLANATION_PROMPT.format(
        question=question,
        data_summary=summary,
    )

    try:
        return _call_llm(prompt, max_tokens=250)
    except Exception as e:
        log.warning(f"Explanation generation failed: {e}")
        return "Results retrieved successfully. See the data table above."


def attempt_correction(question: str, failed_sql: str, error: str) -> str:
    """Step 4: If execution fails, ask LLM to fix the SQL."""
    prompt = CORRECTION_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        question=question,
        failed_sql=failed_sql,
        error_message=error,
    )
    try:
        corrected = _call_llm(prompt, max_tokens=300)
        corrected = corrected.replace("```sql", "").replace("```", "").strip()
        log.info(f"Correction attempt SQL: {corrected[:100]}")
        return corrected
    except Exception as e:
        log.error(f"Correction attempt failed: {e}")
        return failed_sql


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_question(question: str) -> Dict[str, Any]:
    """
    Full NL-to-SQL pipeline.

    Args:
        question: Natural language question from user.

    Returns:
        {
          "question": str,
          "sql": str,
          "results": DataFrame or None,
          "explanation": str,
          "error": str or None,
          "row_count": int,
        }
    """
    log.info(f"Processing question: {question}")

    result = {
        "question": question,
        "sql": "",
        "results": None,
        "explanation": "",
        "error": None,
        "row_count": 0,
    }

    try:
        # Step 1: Generate SQL
        sql = generate_sql(question)
        result["sql"] = sql

        # Step 2: Validate + correct
        sql = validate_and_correct_sql(sql, question)
        result["sql"] = sql

        # Step 3: Execute
        df, error = run_query(sql)

        if error:
            # Step 4: Try to self-correct
            log.warning(f"First execution failed ({error}) — attempting correction")
            corrected_sql = attempt_correction(question, sql, error)
            df, error = run_query(corrected_sql)
            result["sql"] = corrected_sql

        if error:
            result["error"] = error
            result["explanation"] = f"Could not execute query: {error}"
            return result

        result["results"] = df
        result["row_count"] = len(df)

        # Step 5: Explain results
        result["explanation"] = generate_explanation(question, df, sql)

    except Exception as e:
        log.error(f"Pipeline error: {e}")
        result["error"] = str(e)
        result["explanation"] = (
            "An error occurred. Please check your API key configuration "
            "or try rephrasing your question."
        )

    return result
