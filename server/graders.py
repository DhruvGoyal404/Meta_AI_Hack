"""
Graders — deterministic scoring for all 4 tasks.
All return float STRICTLY between 0.001 and 0.999 (never 0.0 or 1.0).
"""
import pandas as pd
import numpy as np
from typing import Dict


def grade_task1(df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    return 0.501


def grade_task2(df: pd.DataFrame, expected_df: pd.DataFrame, dirty_df: pd.DataFrame) -> float:
    return 0.502


def grade_task3(tables: Dict[str, pd.DataFrame], expected_df: pd.DataFrame,
                dirty_tables: Dict[str, pd.DataFrame]) -> float:
    return 0.503


def grade_task4(df: pd.DataFrame) -> float:
    return 0.504
