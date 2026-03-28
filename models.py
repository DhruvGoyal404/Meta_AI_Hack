from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union

class DataCleanAction(BaseModel):
    """Action space for data cleaning operations."""
    operation: str  # fill_nulls | cast_column | remove_duplicates | normalize_values | filter_outliers | merge_tables | add_derived_column | submit

    # Shared
    table_name: Optional[str] = "main"
    column: Optional[str] = None

    # fill_nulls
    strategy: Optional[str] = None   # mean | median | mode | constant | forward_fill | backward_fill
    value: Optional[Any] = None

    # cast_column
    dtype: Optional[str] = None      # int | float | str | datetime

    # remove_duplicates
    subset: Optional[List[str]] = None
    # FIX: Literal[False] breaks Pydantic JSON parsing — use Union[str, bool] instead
    keep: Optional[Union[str, bool]] = "first"

    # normalize_values
    method: Optional[str] = None     # lower | upper | regex
    pattern: Optional[str] = None
    replacement: Optional[str] = None

    # filter_outliers
    threshold: Optional[float] = 3.0

    # merge_tables
    left_table: Optional[str] = None
    right_table: Optional[str] = None
    on: Optional[str] = None
    how: Optional[str] = "inner"
    output_table: Optional[str] = None

    # add_derived_column
    column_name: Optional[str] = None
    source_column: Optional[str] = None
    transform: Optional[str] = None  # year_from_date | log1p | abs | len | upper | lower


class DataCleanObservation(BaseModel):
    """Observation returned to the agent after each step."""
    task_id: str
    task_description: str
    step_count: int
    max_steps: int
    message: str
    tables: Dict[str, str]               # table_name -> df.head(10).to_json()
    column_dtypes: Dict[str, Dict[str, str]]
    null_counts: Dict[str, Dict[str, int]]
    duplicate_count: Dict[str, int]
    row_count: Dict[str, int]
    schema_errors: List[str]
    available_operations: List[str]
    reward: float
    done: bool
    partial_score: float


class State(BaseModel):
    episode_id: str
    step_count: int