"""
Versioned prompt templates for the NL-to-SQL pipeline.

Design principles:
  1. Schema grounding — full column list in every prompt prevents hallucinated columns
  2. Strict output format — "return ONLY the SQL" prevents prose contamination
  3. Safety constraints — blocklist + prompt-level instruction
  4. Separation of concerns — SQL generation, validation, and explanation are different prompts
  5. Version tags — allows A/B testing and rollback

VERSION HISTORY:
  v1.0 — Initial release
  v1.1 — Added DAYS_EMPLOYED anomaly note, stricter LIMIT instruction
  v1.2 — Added JSON validation format, explanation prompt
"""

# ── Schema (injected into prompts) ───────────────────────────────────────────

SCHEMA_DESCRIPTION = """
DATABASE: SQLite
TABLE: applications
DESCRIPTION: Home Credit loan application records. Each row represents one loan applicant.

COLUMNS:
  SK_ID_CURR           INTEGER  — Unique applicant ID (primary key)
  TARGET               INTEGER  — Loan outcome: 1=defaulted, 0=repaid
  NAME_CONTRACT_TYPE   TEXT     — Contract type: 'Cash loans' or 'Revolving loans'
  CODE_GENDER          TEXT     — Gender: 'M' or 'F'
  FLAG_OWN_CAR         TEXT     — Car ownership: 'Y' or 'N'
  FLAG_OWN_REALTY      TEXT     — Property ownership: 'Y' or 'N'
  CNT_CHILDREN         INTEGER  — Number of children
  AMT_INCOME_TOTAL     REAL     — Annual income (USD)
  AMT_CREDIT           REAL     — Loan amount requested (USD)
  AMT_ANNUITY          REAL     — Monthly repayment amount (USD)
  AMT_GOODS_PRICE      REAL     — Price of goods/property being financed (USD)
  NAME_INCOME_TYPE     TEXT     — Employment type (e.g. 'Working', 'Pensioner', 'Commercial associate')
  NAME_EDUCATION_TYPE  TEXT     — Education level (e.g. 'Higher education', 'Secondary / secondary special')
  NAME_FAMILY_STATUS   TEXT     — Marital status (e.g. 'Married', 'Single / not married')
  NAME_HOUSING_TYPE    TEXT     — Housing situation (e.g. 'House / apartment', 'Rented apartment')
  DAYS_BIRTH           INTEGER  — Age in days (ALWAYS NEGATIVE — use CAST(DAYS_BIRTH AS REAL)/-365.0 for years)
  DAYS_EMPLOYED        INTEGER  — Employment duration in days (NEGATIVE if employed; 365243 means NOT EMPLOYED)
  OCCUPATION_TYPE      TEXT     — Job type (e.g. 'Laborers', 'Sales staff', 'Managers')
  CNT_FAM_MEMBERS      REAL     — Total family size including applicant
  REGION_RATING_CLIENT INTEGER  — Region credit risk rating: 1=low, 2=medium, 3=high risk
  EXT_SOURCE_1         REAL     — External credit score 1 (range 0–1; higher = better creditworthiness)
  EXT_SOURCE_2         REAL     — External credit score 2 (range 0–1; higher = better creditworthiness)
  EXT_SOURCE_3         REAL     — External credit score 3 (range 0–1; higher = better creditworthiness)
  CREDIT_INCOME_RATIO  REAL     — Engineered feature: AMT_CREDIT / AMT_INCOME_TOTAL (debt burden)
  ANNUITY_INCOME_RATIO REAL     — Engineered feature: AMT_ANNUITY / AMT_INCOME_TOTAL (monthly payment strain)

IMPORTANT NOTES:
  - DAYS_BIRTH is always negative. To get age: CAST(DAYS_BIRTH AS REAL) / -365.0
  - DAYS_EMPLOYED = 365243 means the applicant has no employment record — exclude with: DAYS_EMPLOYED != 365243
  - TARGET=1 means the loan defaulted (bad outcome); TARGET=0 means it was repaid (good outcome)
  - Default rate = AVG(TARGET) or SUM(TARGET)*100.0/COUNT(*)
""".strip()


# ── Prompt v1.2: SQL Generation ───────────────────────────────────────────────

SQL_GENERATION_PROMPT = """You are a credit risk SQL analyst. Convert the user's natural language question into a valid SQLite SELECT query.

{schema}

STRICT RULES:
1. Return ONLY the SQL query — no explanation, no markdown, no backticks
2. Use ONLY SELECT statements — never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER
3. Reference ONLY columns listed in the schema above — no invented columns
4. Always include LIMIT 500 unless the user asks for all records
5. For percentage calculations use: ROUND(AVG(TARGET)*100.0, 2)
6. For age in years use: ROUND(CAST(DAYS_BIRTH AS REAL) / -365.0, 1)
7. If the question cannot be answered with the schema, return exactly:
   SELECT 'This question cannot be answered with the available data' AS message;

User question: {question}
SQL query:"""


# ── Prompt v1.2: SQL Validation ───────────────────────────────────────────────

SQL_VALIDATION_PROMPT = """Review this SQL query against the credit risk database schema and validate it.

{schema}

SQL TO VALIDATE:
{sql}

Check:
1. Is it a SELECT statement only? (not INSERT/UPDATE/DELETE/DROP/CREATE)
2. Does it use only columns that exist in the schema?
3. Is it valid SQLite syntax?
4. Does it have a reasonable LIMIT?
5. Are DAYS_BIRTH and DAYS_EMPLOYED used correctly?

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"valid": true, "issues": [], "corrected_sql": "{sql}"}}

Or if there are issues:
{{"valid": false, "issues": ["describe each issue"], "corrected_sql": "the fixed SQL here"}}"""


# ── Prompt v1.2: Business Explanation ─────────────────────────────────────────

EXPLANATION_PROMPT = """You are a credit risk business analyst. Explain query results in plain English.

User asked: "{question}"

Query result summary:
{data_summary}

Write 2-3 sentences explaining:
- What the data shows
- What it means for credit risk management
- Any actionable insight

Use plain business language. No SQL. No technical jargon. No bullet points."""


# ── Prompt: Correction (when execution fails) ────────────────────────────────

CORRECTION_PROMPT = """The SQL query failed with an error. Fix it.

{schema}

Original question: {question}
Failed SQL: {failed_sql}
Error message: {error_message}

Return ONLY the corrected SQL query. No explanation."""


# ── Demo queries (5 required by assignment) ──────────────────────────────────

DEMO_QUERIES = [
    {
        "label": "Default rate by education level",
        "question": "What is the default rate for each education level?",
    },
    {
        "label": "Average loan by contract type",
        "question": "What is the average loan amount for Cash loans vs Revolving loans?",
    },
    {
        "label": "Income type with highest default",
        "question": "Which income type has the highest default rate?",
    },
    {
        "label": "Top defaulted applicants",
        "question": "Show me the top 10 applicants with the highest credit amounts who defaulted",
    },
    {
        "label": "Gender and default analysis",
        "question": "What is the default rate by gender and do women or men default more?",
    },
    {
        "label": "Region risk distribution",
        "question": "How many applicants are in each region risk rating and what is the default rate per region?",
    },
    {
        "label": "Age group default analysis",
        "question": "What is the default rate for applicants grouped by age: under 30, 30-45, 45-60, and over 60?",
    },
    {
        "label": "Property ownership impact",
        "question": "Do applicants who own property have a lower default rate than those who don't?",
    },
]
