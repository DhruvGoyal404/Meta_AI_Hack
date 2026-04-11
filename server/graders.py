# """
# Graders — deterministic scoring for all 4 tasks.
# All return float STRICTLY between 0.001 and 0.999 (never 0.0 or 1.0).
# """
# import pandas as pd
# import numpy as np
# from typing import Dict


# def grade_task1(df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
#     return 0.501


# def grade_task2(df: pd.DataFrame, expected_df: pd.DataFrame, dirty_df: pd.DataFrame) -> float:
#     return 0.502


# def grade_task3(tables: Dict[str, pd.DataFrame], expected_df: pd.DataFrame,
#                 dirty_tables: Dict[str, pd.DataFrame]) -> float:
#     return 0.503


# def grade_task4(df: pd.DataFrame) -> float:
#     return 0.504

"""
Graders — deterministic partial-credit scoring for all 4 tasks.
Every grader returns a float STRICTLY in (0.0, 1.0).
  • Never returns exactly 0.0 — even a completely uncleaned table scores 0.05.
  • Never returns exactly 1.0 — a perfect table scores 0.98.
  • All intermediate states return a meaningful float between those bounds.

This satisfies the OpenEnv validator requirement:
  "one or more tasks returned a score outside [0, 1]"  ← was caused by hardcoded stubs.

Grader design:
  - Each grader checks multiple sub-dimensions with individual weights.
  - Weights sum to 1.0 for each grader.
  - Raw score is clipped to [0.05, 0.98] before return.
"""
import pandas as pd
import numpy as np
from typing import Dict


def _clamp(score: float) -> float:
    """Clamp to strictly-open (0, 1) range required by OpenEnv validator."""
    return float(max(0.05, min(0.98, score)))


# ── Task 1 ────────────────────────────────────────────────────────────────────

def grade_task1(df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    """
    Score a task1 'main' DataFrame.

    Sub-dimensions (weights sum to 1.0):
      - age nulls gone          0.30
      - age dtype is int64      0.25
      - age values close to expected  0.20
      - salary nulls gone       0.15
      - salary dtype is float64 0.10
    """
    score = 0.0

    # age nulls (0.30)
    age_nulls = int(df["age"].isna().sum()) if "age" in df.columns else 999
    if age_nulls == 0:
        score += 0.30
    elif age_nulls <= 3:
        score += 0.15

    # age dtype (0.25)
    if "age" in df.columns:
        age_numeric = pd.to_numeric(df["age"], errors="coerce")
        non_null_age = age_numeric.dropna()
        if pd.api.types.is_integer_dtype(df["age"]):
            score += 0.25
        elif len(non_null_age) == len(df) and (non_null_age % 1 == 0).all():
            # numeric but stored as float with no decimals — partial credit
            score += 0.12

    # age values accuracy (0.20) — compare median of cleaned vs expected
    if "age" in df.columns and "age" in expected_df.columns:
        try:
            actual_med = pd.to_numeric(df["age"], errors="coerce").median()
            exp_med = pd.to_numeric(expected_df["age"], errors="coerce").median()
            if abs(actual_med - exp_med) < 1:
                score += 0.20
            elif abs(actual_med - exp_med) < 5:
                score += 0.10
        except Exception:
            pass

    # salary nulls (0.15)
    sal_nulls = int(df["salary"].isna().sum()) if "salary" in df.columns else 999
    if sal_nulls == 0:
        score += 0.15
    elif sal_nulls <= 3:
        score += 0.07

    # salary dtype (0.10)
    if "salary" in df.columns and pd.api.types.is_float_dtype(df["salary"]):
        score += 0.10

    return _clamp(score)


# ── Task 2 ────────────────────────────────────────────────────────────────────

def grade_task2(df: pd.DataFrame, expected_df: pd.DataFrame,
                dirty_df: pd.DataFrame) -> float:
    """
    Score a task2 'main' DataFrame.

    Sub-dimensions (weights sum to 1.0):
      - duplicates removed      0.25
      - country normalised      0.25
      - order_date is datetime  0.20
      - amount nulls gone       0.15
      - row count reasonable    0.15
    """
    score = 0.0

    # duplicates (0.25)
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        score += 0.25
    elif dup_count < 5:
        score += 0.12

    # country upper-case (0.25)
    if "country" in df.columns:
        str_col = df["country"].dropna().astype(str)
        total = len(str_col)
        if total > 0:
            upper_frac = (str_col == str_col.str.upper()).mean()
            score += 0.25 * upper_frac

    # order_date datetime (0.20)
    if "order_date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            score += 0.20
        else:
            parsed = pd.to_datetime(df["order_date"], errors="coerce")
            valid_frac = parsed.notna().mean()
            score += 0.20 * valid_frac * 0.5  # partial: parsable but wrong dtype

    # amount nulls (0.15)
    if "amount" in df.columns:
        amt_nulls = int(df["amount"].isna().sum())
        if amt_nulls == 0:
            score += 0.15
        elif amt_nulls <= 3:
            score += 0.07

    # row count (0.15) — should be ≈170 (base) after dedup, not 200 (with dups)
    n_rows = len(df)
    dirty_rows = len(dirty_df)
    expected_rows = len(expected_df)
    if expected_rows > 0:
        ratio = n_rows / expected_rows
        if 0.85 <= ratio <= 1.15:
            score += 0.15
        elif 0.5 <= ratio <= 1.5:
            score += 0.07

    return _clamp(score)


# ── Task 3 ────────────────────────────────────────────────────────────────────

def grade_task3(tables: Dict[str, pd.DataFrame], expected_df: pd.DataFrame,
                dirty_tables: Dict[str, pd.DataFrame]) -> float:
    """
    Score a task3 state.

    Sub-dimensions (weights sum to 1.0):
      - merged table exists            0.25
      - outliers removed (amount IQR)  0.25
      - age nulls gone in merged       0.20
      - order_year column present      0.15
      - row count in reasonable range  0.15
    """
    score = 0.0

    # merged table (0.25)
    merged_key = None
    for k in ("merged", "main"):
        if k in tables:
            merged_key = k
            break
    if merged_key is None:
        # No merge done yet — return base score
        return _clamp(0.05)

    merged = tables[merged_key]

    score += 0.25  # merged table exists

    # outliers removed (0.25) — check that extreme amounts are gone
    if "amount" in merged.columns:
        amt = pd.to_numeric(merged["amount"], errors="coerce").dropna()
        if len(amt) > 0:
            Q1, Q3 = amt.quantile(0.25), amt.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_frac = ((amt < lo) | (amt > hi)).mean()
            if outlier_frac < 0.02:
                score += 0.25
            elif outlier_frac < 0.10:
                score += 0.12

    # age nulls (0.20)
    if "age" in merged.columns:
        age_nulls = int(merged["age"].isna().sum())
        if age_nulls == 0:
            score += 0.20
        elif age_nulls <= 3:
            score += 0.10

    # order_year column (0.15)
    if "order_year" in merged.columns:
        yr = pd.to_numeric(merged["order_year"], errors="coerce")
        valid_years = yr.between(2020, 2030).mean()
        score += 0.15 * valid_years

    # row count (0.15)
    exp_rows = len(expected_df)
    if exp_rows > 0:
        ratio = len(merged) / exp_rows
        if 0.80 <= ratio <= 1.20:
            score += 0.15
        elif 0.50 <= ratio <= 1.50:
            score += 0.07

    return _clamp(score)


# ── Task 4 ────────────────────────────────────────────────────────────────────

def grade_task4(df: pd.DataFrame) -> float:
    """
    Score a task4 'stream' DataFrame.

    Since this task has continuous drift, we score the current *cleaned* state
    of the stream table across multiple dimensions.

    Sub-dimensions (weights sum to 1.0):
      - amount nulls low        0.25
      - amount dtype numeric    0.20
      - outliers low            0.20
      - category nulls low      0.15
      - region nulls low        0.10
      - event_ts parseable      0.10
    """
    score = 0.0
    n = len(df)
    if n == 0:
        return _clamp(0.05)

    # amount nulls (0.25)
    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce")
        null_frac = amt.isna().mean()
        score += 0.25 * max(0.0, 1.0 - null_frac * 3)

        # amount dtype numeric (0.20)
        if pd.api.types.is_numeric_dtype(df["amount"]):
            score += 0.20
        elif null_frac < 0.10:
            # mostly parseable even if still object
            score += 0.10

        # outliers (0.20) — negative or huge values
        valid_amt = amt.dropna()
        if len(valid_amt) > 0:
            Q1, Q3 = valid_amt.quantile(0.25), valid_amt.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_frac = ((valid_amt < lo) | (valid_amt > hi)).mean()
            if outlier_frac < 0.03:
                score += 0.20
            elif outlier_frac < 0.15:
                score += 0.10

    # category nulls (0.15)
    if "category" in df.columns:
        cat_null_frac = df["category"].isna().mean()
        score += 0.15 * max(0.0, 1.0 - cat_null_frac * 3)

    # region nulls (0.10)
    if "region" in df.columns:
        reg_null_frac = df["region"].isna().mean()
        score += 0.10 * max(0.0, 1.0 - reg_null_frac * 3)

    # event_ts parseable (0.10)
    if "event_ts" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["event_ts"]):
            score += 0.10
        else:
            parsed = pd.to_datetime(df["event_ts"], errors="coerce")
            score += 0.10 * parsed.notna().mean()

    return _clamp(score)
