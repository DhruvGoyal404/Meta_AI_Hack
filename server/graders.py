"""
Graders — deterministic scoring for all 4 tasks.
All return float STRICTLY between 0.001 and 0.999 (never 0.0 or 1.0).
"""
import pandas as pd
import numpy as np
from typing import Dict


def _clamp(score: float) -> float:
    """Single source of truth — every grader exits through here.
    Also guards against NaN / inf which would silently pass min/max."""
    try:
        v = float(score)
        if not np.isfinite(v):   # NaN, +inf, -inf → floor
            v = 0.001
        return round(min(0.999, max(0.001, v)), 4)
    except Exception:
        return 0.001


def grade_task1(df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    try:
        if df is None or df.empty:
            return 0.001
        n = max(1, len(df))
        score = 0.0   # FIX: was 0.001 → perfect run gave 1.001 before clamp

        if "age" in df.columns:
            if pd.api.types.is_integer_dtype(df["age"]):
                score += 0.30
            null_r = float(df["age"].isna().sum()) / n
            score += 0.35 * (1.0 - null_r)

        if "salary" in df.columns:
            if pd.api.types.is_float_dtype(df["salary"]) or pd.api.types.is_integer_dtype(df["salary"]):
                score += 0.15
            null_r = float(df["salary"].isna().sum()) / n
            score += 0.20 * (1.0 - null_r)

        return _clamp(score)
    except Exception:
        return 0.001


def grade_task2(df: pd.DataFrame, expected_df: pd.DataFrame, dirty_df: pd.DataFrame) -> float:
    try:
        if df is None or df.empty:
            return 0.001
        n = max(1, len(df))
        score = 0.0   # FIX: was 0.001

        n_dirty_dups = int(dirty_df.duplicated().sum())
        n_curr_dups  = int(df.duplicated().sum())
        if n_dirty_dups > 0:
            # FIX: cap ratio at 1.0 — if agent introduces NEW duplicates the
            # ratio > 1 made the term negative, fighting the max(0.0) guard
            # and producing unexpected sub-floor scores downstream.
            ratio = min(1.0, n_curr_dups / n_dirty_dups)
            score += 0.25 * (1.0 - ratio)
        else:
            score += 0.25

        if "country" in df.columns:
            col = df["country"].astype(str).str.strip()
            score += 0.35 * float((col == col.str.upper()).mean())

        if "order_date" in df.columns:
            parsed = pd.to_datetime(df["order_date"], errors="coerce")
            score += 0.25 * float(parsed.notna().mean())

        if "amount" in df.columns:
            score += 0.15 * (1.0 - float(df["amount"].isna().sum()) / n)

        return _clamp(score)
    except Exception:
        return 0.001


def grade_task3(tables: Dict[str, pd.DataFrame], expected_df: pd.DataFrame,
                dirty_tables: Dict[str, pd.DataFrame]) -> float:
    try:
        df = tables.get("merged")
        if df is None:
            df = tables.get("main")
        if df is None or df.empty:
            return 0.001

        n = max(1, len(df))
        score = 0.0   # FIX: was 0.001

        required_cols = {"order_id", "customer_id", "amount", "name", "country"}
        col_score = len(required_cols & set(df.columns)) / len(required_cols)

        n_orders = len(dirty_tables.get("orders", pd.DataFrame()))
        n_exp    = len(expected_df) if expected_df is not None else 0
        if n_exp > 0:
            row_score = max(0.0, 1.0 - abs(n - n_exp) / max(1, n_exp))
        elif n_orders > 0:
            row_score = min(1.0, n / n_orders)
        else:
            row_score = 1.0

        score += 0.30 * (0.6 * col_score + 0.4 * row_score)

        if "amount" in df.columns:
            amounts = pd.to_numeric(df["amount"], errors="coerce").dropna()
            if len(amounts) > 10:
                Q1, Q3 = amounts.quantile(0.25), amounts.quantile(0.75)
                IQR = Q3 - Q1
                # FIX: guard IQR==0 (all identical amounts) → division artefact
                if IQR > 0:
                    outlier_ratio = float(
                        ((amounts < Q1 - 1.5 * IQR) | (amounts > Q3 + 1.5 * IQR)).mean()
                    )
                else:
                    outlier_ratio = 0.0
                score += 0.25 * max(0.0, 1.0 - outlier_ratio * 10)
            else:
                score += 0.05

        dtype_checks = [
            ("customer_id", pd.api.types.is_integer_dtype),
            ("order_id",    pd.api.types.is_integer_dtype),
            ("amount",      lambda c: pd.api.types.is_float_dtype(c) or pd.api.types.is_integer_dtype(c)),
            ("name",        lambda c: pd.api.types.is_object_dtype(c) or pd.api.types.is_string_dtype(c)),
        ]
        hits = sum(
            1 for col, chk in dtype_checks
            if col in df.columns and chk(df[col])
        )
        score += 0.25 * (hits / len(dtype_checks))

        if "order_year" in df.columns:
            years = pd.to_numeric(df["order_year"], errors="coerce")
            valid = float(((years >= 2020) & (years <= 2030)).mean())
            score += 0.20 * valid

        return _clamp(score)
    except Exception:
        return 0.001


def grade_task4(df: pd.DataFrame) -> float:
    try:
        if df is None or df.empty:
            return 0.001

        n = max(1, len(df))
        score = 0.0   # FIX: was 0.001

        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce")
            null_r = float(amt.isna().sum()) / n
            if null_r < 1.0:
                Q1, Q3 = amt.dropna().quantile(0.25), amt.dropna().quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outlier_r = float(((amt < Q1 - 1.5*IQR) | (amt > Q3 + 1.5*IQR)).mean())
                else:
                    outlier_r = 0.0
            else:
                outlier_r = 1.0
            amt_score = (1.0 - null_r) * 0.5 + (1.0 - min(1.0, outlier_r * 5)) * 0.5
            score += 0.30 * max(0.0, amt_score)

        if "event_ts" in df.columns:
            parsed = pd.to_datetime(df["event_ts"], errors="coerce")
            score += 0.25 * float(parsed.notna().mean())

        cat_ok = (1.0 - float(df["category"].isna().sum()) / n) if "category" in df.columns else 0.0
        reg_ok = (1.0 - float(df["region"].isna().sum()) / n)   if "region"   in df.columns else 0.0
        score += 0.25 * (cat_ok * 0.5 + reg_ok * 0.5)

        # FIX: was min(0.999, n/60.0) — should cap at 1.0 so the weight
        # coefficient 0.20 is applied correctly. _clamp() enforces 0.999 ceiling
        # on the TOTAL score, not on individual components.
        row_health = min(1.0, n / 60.0)
        score += 0.20 * row_health

        return _clamp(score)
    except Exception:
        return 0.001
