-- ============================================================
-- Credit Risk Platform — SQLite Schema
-- ============================================================
-- This schema documents the structure of the 'applications' table
-- which is populated at runtime from application_train.csv.
-- The table is created by src/data/loader.py using pandas .to_sql()
-- ============================================================

CREATE TABLE IF NOT EXISTS applications (
    SK_ID_CURR           INTEGER PRIMARY KEY,
    TARGET               INTEGER,          -- 1=defaulted, 0=repaid
    NAME_CONTRACT_TYPE   TEXT,
    CODE_GENDER          TEXT,
    FLAG_OWN_CAR         TEXT,
    FLAG_OWN_REALTY      TEXT,
    CNT_CHILDREN         INTEGER,
    AMT_INCOME_TOTAL     REAL,
    AMT_CREDIT           REAL,
    AMT_ANNUITY          REAL,
    AMT_GOODS_PRICE      REAL,
    NAME_INCOME_TYPE     TEXT,
    NAME_EDUCATION_TYPE  TEXT,
    NAME_FAMILY_STATUS   TEXT,
    NAME_HOUSING_TYPE    TEXT,
    DAYS_BIRTH           INTEGER,
    DAYS_EMPLOYED        INTEGER,
    OCCUPATION_TYPE      TEXT,
    CNT_FAM_MEMBERS      REAL,
    REGION_RATING_CLIENT INTEGER,
    EXT_SOURCE_1         REAL,
    EXT_SOURCE_2         REAL,
    EXT_SOURCE_3         REAL,
    CREDIT_INCOME_RATIO  REAL,
    ANNUITY_INCOME_RATIO REAL
);

-- ── Sample queries for reference / testing ────────────────────────────────

-- Default rate overall
-- SELECT ROUND(AVG(TARGET)*100.0, 2) AS default_rate_pct FROM applications;

-- Default rate by education
-- SELECT NAME_EDUCATION_TYPE, ROUND(AVG(TARGET)*100.0,2) AS default_pct, COUNT(*) AS n
-- FROM applications GROUP BY NAME_EDUCATION_TYPE ORDER BY default_pct DESC;

-- Age group analysis
-- SELECT
--   CASE
--     WHEN CAST(DAYS_BIRTH AS REAL)/-365.0 < 30 THEN 'Under 30'
--     WHEN CAST(DAYS_BIRTH AS REAL)/-365.0 < 45 THEN '30-45'
--     WHEN CAST(DAYS_BIRTH AS REAL)/-365.0 < 60 THEN '45-60'
--     ELSE 'Over 60'
--   END AS age_group,
--   ROUND(AVG(TARGET)*100.0,2) AS default_pct,
--   COUNT(*) AS n
-- FROM applications
-- GROUP BY age_group;
