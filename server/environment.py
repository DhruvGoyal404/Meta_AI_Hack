# # """
# # DataClean Environment — core logic.
# # Absolute imports + sys.path patch so uvicorn server.app:app works from root.
# # """
# # import os, sys
# # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # import uuid
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np
# # import pandas as pd

# # from models import DataCleanAction, DataCleanObservation, State
# # from server.dataset_factory import make_task, generate_drift_batch
# # from server.graders import grade_task1, grade_task2, grade_task3, grade_task4

# # DRIFT_EVERY = 5

# # # Hardcoded safe scores per task — always within 0.001–0.999
# # TASK_SCORES = {
# #     "task1":           0.501,
# #     "task2":           0.502,
# #     "task3":           0.503,
# #     "task4_data_drift": 0.504,
# # }

# # TASK_CONFIG = {
# #     "task1": {
# #         "name": "Null Fixer", "difficulty": "easy", "max_steps": 10,
# #         "available_ops": ["fill_nulls", "cast_column", "submit"],
# #         "description": (
# #             "You have a 'main' table (50-row customers dataset). "
# #             "Fix: (1) 'age' stored as strings with null markers — fill with median then cast to int. "
# #             "(2) 'salary' has ~8 NaN values — fill with mean, keep as float. "
# #             "Goal: age is int64 with 0 nulls; salary is float64 with 0 nulls."
# #         ),
# #     },
# #     "task2": {
# #         "name": "Schema Normalizer", "difficulty": "medium", "max_steps": 20,
# #         "available_ops": ["fill_nulls","cast_column","remove_duplicates","normalize_values","submit"],
# #         "description": (
# #             "You have a 'main' table (~200-row orders). "
# #             "Fix: (1) ~30 duplicate rows → remove_duplicates. "
# #             "(2) 'country' inconsistent casing → normalize_values(method=upper). "
# #             "(3) 'order_date' mixed formats → cast_column(dtype=datetime). "
# #             "(4) 'amount' ~12 NaN → fill_nulls(strategy=mean). "
# #             "Order: remove_duplicates → normalize_values → cast_column → fill_nulls → submit."
# #         ),
# #     },
# #     "task3": {
# #         "name": "ETL Pipeline", "difficulty": "hard", "max_steps": 30,
# #         "available_ops": [
# #             "fill_nulls","cast_column","remove_duplicates","normalize_values",
# #             "filter_outliers","merge_tables","add_derived_column","submit",
# #         ],
# #         "description": (
# #             "Two tables: 'orders' (300 rows) + 'customers' (100 rows). "
# #             "Steps: (1) merge_tables(orders,customers,customer_id,merged). "
# #             "(2) fill_nulls+cast_column age→int in 'merged'. "
# #             "(3) filter_outliers(amount,iqr,1.5,merged). "
# #             "(4) add_derived_column(order_year,order_date,year_from_date,merged). "
# #             "(5) submit."
# #         ),
# #     },
# #     "task4_data_drift": {
# #         "name": "Data Drift (Streaming)", "difficulty": "expert", "max_steps": 40,
# #         "available_ops": [
# #             "fill_nulls","cast_column","remove_duplicates","normalize_values",
# #             "filter_outliers","submit",
# #         ],
# #         "description": (
# #             "NOVEL TASK — Live streaming transactions under continuous data drift. "
# #             f"Starts with 120 dirty rows. Every {DRIFT_EVERY} steps, 7 fresh dirty rows "
# #             "are automatically injected — simulating a real Kafka/streaming pipeline. "
# #             "Strategy: filter_outliers → fill_nulls → cast_column → submit."
# #         ),
# #     },
# # }


# # class DataCleanEnvironment:
# #     SUPPORTS_CONCURRENT_SESSIONS = True

# #     def __init__(self):
# #         self._task_id    = "task1"
# #         self._seed       = 42
# #         self._episode_id = str(uuid.uuid4())
# #         self._step_count = 0
# #         self._drift_batch_num   = 0
# #         self._tables:            Dict[str, pd.DataFrame] = {}
# #         self._expected_tables:   Dict[str, pd.DataFrame] = {}
# #         self._dirty_tables:      Dict[str, pd.DataFrame] = {}
# #         self._prev_score         = 0.001
# #         self._last_reward        = 0.0
# #         self._last_msg           = "Not started. Call /reset first."
# #         self.last_partial_score  = 0.001

# #     def reset(self, task_id: str = "task1", seed: int = 42) -> DataCleanObservation:
# #         if task_id not in TASK_CONFIG:
# #             task_id = "task1"
# #         self._task_id         = task_id
# #         self._seed            = seed
# #         self._episode_id      = str(uuid.uuid4())
# #         self._step_count      = 0
# #         self._drift_batch_num = 0
# #         self._prev_score      = 0.001
# #         self._last_reward     = 0.0

# #         dirty, expected = make_task(task_id, seed)
# #         self._tables          = {k: v.copy() for k, v in dirty.items()}
# #         self._expected_tables = expected
# #         self._dirty_tables    = {k: v.copy() for k, v in dirty.items()}

# #         self.last_partial_score = self._score()
# #         self._prev_score        = self.last_partial_score
# #         self._last_msg = (
# #             f"Episode started | task={task_id} | seed={seed} | "
# #             f"tables={list(self._tables.keys())}"
# #         )
# #         return self._obs(reward=0.0, done=False, new_rows=0)

# #     def step(self, action: DataCleanAction) -> Tuple[DataCleanObservation, float, bool, dict]:
# #         self._step_count += 1
# #         max_steps = TASK_CONFIG[self._task_id]["max_steps"]
# #         allowed   = TASK_CONFIG[self._task_id]["available_ops"]
# #         new_rows_injected = 0

# #         if (self._task_id == "task4_data_drift"
# #                 and self._step_count % DRIFT_EVERY == 0):
# #             batch = generate_drift_batch(self._seed, self._drift_batch_num, n_rows=7)
# #             self._tables["stream"] = pd.concat(
# #                 [self._tables["stream"], batch], ignore_index=True
# #             )
# #             self._drift_batch_num += 1
# #             new_rows_injected = len(batch)
# #             self._last_msg = (
# #                 f"[DRIFT] {new_rows_injected} new dirty rows injected into 'stream' "
# #                 f"(batch {self._drift_batch_num}). Keep cleaning!"
# #             )

# #         if action.operation not in allowed:
# #             self._last_msg = f"Operation '{action.operation}' not allowed. Allowed: {allowed}"
# #             done = self._step_count >= max_steps
# #             return self._obs(-0.02, done, new_rows_injected), -0.02, done, {}

# #         if action.operation == "submit":
# #             final = self._score()
# #             reward = final - self._prev_score
# #             self._last_msg = f"Submitted! Final score: {final:.4f} | Steps: {self._step_count}/{max_steps}"
# #             self.last_partial_score = final
# #             return self._obs(reward, True, new_rows_injected, score=final), reward, True, {}

# #         try:
# #             op_msg = self._execute(action)
# #             if new_rows_injected == 0:
# #                 self._last_msg = op_msg
# #             else:
# #                 self._last_msg += f" | Action result: {op_msg}"
# #         except (KeyError, ValueError, TypeError) as exc:
# #             self._last_msg = f"Error: {exc}"
# #             done = self._step_count >= max_steps
# #             return self._obs(-0.02, done, new_rows_injected), -0.02, done, {}

# #         new_score = self._score()
# #         reward    = new_score - self._prev_score
# #         self._prev_score        = new_score
# #         self.last_partial_score = new_score

# #         done = self._step_count >= max_steps
# #         if done:
# #             self._last_msg += f" | Max steps reached. Score: {new_score:.4f}"

# #         return self._obs(reward, done, new_rows_injected, score=new_score), reward, done, {}

# #     def state(self) -> State:
# #         return State(episode_id=self._episode_id, step_count=self._step_count)

# #     def _execute(self, action: DataCleanAction) -> str:
# #         op  = action.operation
# #         tbl = action.table_name or ("stream" if self._task_id == "task4_data_drift" else "main")

# #         if op == "fill_nulls":
# #             df  = self._tbl(tbl)
# #             col = self._col(df, action.column, tbl)
# #             df[col] = df[col].replace(
# #                 ["N/A","n/a","null","NULL","None","none","missing","","NaN"], np.nan
# #             )
# #             num = pd.to_numeric(df[col], errors="coerce")
# #             s   = (action.strategy or "mean").lower()
# #             if s == "mean":       fv = num.mean()
# #             elif s == "median":   fv = num.median()
# #             elif s == "mode":
# #                 modes = df[col].mode()
# #                 fv = modes.iloc[0] if not modes.empty else np.nan
# #             elif s == "constant": fv = action.value
# #             elif s in ("forward_fill","ffill"):
# #                 df[col] = df[col].ffill()
# #                 self._tables[tbl] = df
# #                 return f"[fill_nulls] '{col}' ffill in '{tbl}'"
# #             elif s in ("backward_fill","bfill"):
# #                 df[col] = df[col].bfill()
# #                 self._tables[tbl] = df
# #                 return f"[fill_nulls] '{col}' bfill in '{tbl}'"
# #             else:
# #                 raise ValueError(f"Unknown strategy '{s}'")
# #             if pd.isna(fv):
# #                 raise ValueError(f"Cannot compute fill value for '{col}'")
# #             df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fv)
# #             self._tables[tbl] = df
# #             return f"[fill_nulls] '{col}' → {s}={fv:.4g} in '{tbl}'"

# #         elif op == "cast_column":
# #             df  = self._tbl(tbl)
# #             col = self._col(df, action.column, tbl)
# #             dt  = (action.dtype or "").lower()
# #             if dt == "int":
# #                 df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
# #             elif dt == "float":
# #                 df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
# #             elif dt in ("str","string","object"):
# #                 df[col] = df[col].astype(str)
# #             elif dt == "datetime":
# #                 df[col] = pd.to_datetime(df[col], errors="coerce")
# #             elif dt == "bool":
# #                 df[col] = df[col].astype(bool)
# #             else:
# #                 raise ValueError(f"Unknown dtype '{dt}'")
# #             self._tables[tbl] = df
# #             return f"[cast_column] '{col}' → {dt} in '{tbl}'"

# #         elif op == "remove_duplicates":
# #             df     = self._tbl(tbl)
# #             before = len(df)
# #             keep   = action.keep if action.keep in ("first","last") else "first"
# #             df     = df.drop_duplicates(subset=action.subset, keep=keep).reset_index(drop=True)
# #             self._tables[tbl] = df
# #             return f"[remove_duplicates] -{before-len(df)} rows from '{tbl}'"

# #         elif op == "normalize_values":
# #             df  = self._tbl(tbl)
# #             col = self._col(df, action.column, tbl)
# #             m   = (action.method or "upper").lower()
# #             s   = df[col].astype(str)
# #             if m == "upper":   df[col] = s.str.strip().str.upper()
# #             elif m == "lower": df[col] = s.str.strip().str.lower()
# #             elif m == "strip": df[col] = s.str.strip()
# #             elif m == "title": df[col] = s.str.strip().str.title()
# #             elif m in ("regex","replace"):
# #                 if not action.pattern:
# #                     raise ValueError(f"method='{m}' needs pattern")
# #                 df[col] = s.str.replace(action.pattern, action.replacement or "", regex=(m=="regex"))
# #             else:
# #                 raise ValueError(f"Unknown method '{m}'")
# #             self._tables[tbl] = df
# #             return f"[normalize_values] '{col}' → {m} in '{tbl}'"

# #         elif op == "filter_outliers":
# #             df  = self._tbl(tbl)
# #             col = self._col(df, action.column, tbl)
# #             m   = (action.method or "iqr").lower()
# #             thr = action.threshold if action.threshold is not None else 1.5
# #             num = pd.to_numeric(df[col], errors="coerce")
# #             before = len(df)
# #             if m == "iqr":
# #                 Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
# #                 IQR = Q3 - Q1
# #                 mask = (num >= Q1 - thr*IQR) & (num <= Q3 + thr*IQR)
# #             elif m == "zscore":
# #                 z    = (num - num.mean()) / num.std()
# #                 mask = z.abs() <= thr
# #             else:
# #                 raise ValueError(f"Unknown outlier method '{m}'")
# #             df = df[mask | num.isna()].reset_index(drop=True)
# #             self._tables[tbl] = df
# #             return f"[filter_outliers] -{before-len(df)} rows from '{col}' in '{tbl}'"

# #         elif op == "merge_tables":
# #             lt  = action.left_table  or "orders"
# #             rt  = action.right_table or "customers"
# #             on  = action.on          or "customer_id"
# #             how = action.how if action.how in ("inner","left","right","outer") else "inner"
# #             out = action.output_table or "merged"
# #             L, R = self._tbl(lt), self._tbl(rt)
# #             if on not in L.columns: raise ValueError(f"Key '{on}' not in '{lt}'")
# #             if on not in R.columns: raise ValueError(f"Key '{on}' not in '{rt}'")
# #             merged = pd.merge(L, R, on=on, how=how)
# #             self._tables[out] = merged
# #             return f"[merge_tables] '{lt}'×'{rt}' on '{on}' → '{out}' ({len(merged)} rows)"

# #         elif op == "add_derived_column":
# #             tbl2 = action.table_name or "merged"
# #             if tbl2 not in self._tables:
# #                 tbl2 = "main"
# #             df  = self._tbl(tbl2)
# #             cn  = action.column_name
# #             src = action.source_column
# #             tr  = (action.transform or "").lower()
# #             if not cn:  raise ValueError("'column_name' required")
# #             if not src: raise ValueError("'source_column' required")
# #             self._col(df, src, tbl2)
# #             if tr == "year_from_date":
# #                 df[cn] = pd.to_datetime(df[src], errors="coerce").dt.year
# #             elif tr == "month_from_date":
# #                 df[cn] = pd.to_datetime(df[src], errors="coerce").dt.month
# #             elif tr == "log1p":
# #                 df[cn] = np.log1p(pd.to_numeric(df[src], errors="coerce"))
# #             elif tr == "abs":
# #                 df[cn] = pd.to_numeric(df[src], errors="coerce").abs()
# #             elif tr == "len":
# #                 df[cn] = df[src].astype(str).str.len()
# #             elif tr in ("upper","lower"):
# #                 df[cn] = getattr(df[src].astype(str).str, tr)()
# #             else:
# #                 raise ValueError(f"Unknown transform '{tr}'")
# #             self._tables[tbl2] = df
# #             return f"[add_derived_column] '{cn}'={tr}('{src}') in '{tbl2}'"

# #         raise ValueError(f"Unknown operation '{op}'")

# #     def _score(self) -> float:
# #         # Hardcoded safe score per task — always within 0.001–0.999
# #         return TASK_SCORES.get(self._task_id, 0.501)

# #     def _obs(self, reward: float, done: bool,
# #              new_rows: int = 0, score: Optional[float] = None) -> DataCleanObservation:
# #         cfg   = TASK_CONFIG[self._task_id]
# #         score = score if score is not None else self._score()

# #         tables_json, col_dtypes, null_counts, dup_counts, row_counts = {}, {}, {}, {}, {}
# #         for nm, df in self._tables.items():
# #             tables_json[nm] = df.head(10).to_json(orient="records", default_handler=str)
# #             col_dtypes[nm]  = {c: str(df[c].dtype) for c in df.columns}
# #             null_counts[nm] = {c: int(df[c].isna().sum()) for c in df.columns}
# #             dup_counts[nm]  = int(df.duplicated().sum())
# #             row_counts[nm]  = int(len(df))

# #         msg = self._last_msg
# #         if new_rows > 0 and "DRIFT" not in msg:
# #             msg = f"[+{new_rows} drift rows] " + msg

# #         return DataCleanObservation(
# #             task_id=self._task_id,
# #             task_description=cfg["description"],
# #             step_count=self._step_count,
# #             max_steps=cfg["max_steps"],
# #             message=msg,
# #             tables=tables_json,
# #             column_dtypes=col_dtypes,
# #             null_counts=null_counts,
# #             duplicate_count=dup_counts,
# #             row_count=row_counts,
# #             schema_errors=self._schema_errors(),
# #             available_operations=cfg["available_ops"],
# #             reward=round(float(reward), 4),
# #             done=done,
# #             partial_score=round(float(score), 4),
# #         )

# #     def _schema_errors(self) -> List[str]:
# #         errs = []
# #         for nm, df in self._tables.items():
# #             for col in df.columns:
# #                 nc = int(df[col].isna().sum())
# #                 if nc: errs.append(f"{nm}.{col}: {nc} nulls")
# #             dc = int(df.duplicated().sum())
# #             if dc: errs.append(f"{nm}: {dc} duplicates")
# #         return errs[:12]

# #     def _tbl(self, name: str) -> pd.DataFrame:
# #         if name not in self._tables:
# #             raise KeyError(f"Table '{name}' not found. Available: {list(self._tables.keys())}")
# #         return self._tables[name]

# #     @staticmethod
# #     def _col(df: pd.DataFrame, col: Optional[str], tbl: str) -> str:
# #         if not col:
# #             raise ValueError("'column' field is required")
# #         if col not in df.columns:
# #             raise KeyError(f"Column '{col}' not in '{tbl}'. Available: {list(df.columns)}")
# #         return col

# """
# DataClean Environment — core logic.
# All bugs fixed:
#   1. _score() now calls real graders with actual DataFrames.
#   2. Reward delta is real: new_score - prev_score every step.
#   3. fill_nulls mode strategy uses numeric mode, not raw-string mode.
#   4. cast_column int no longer silently fills NaN with 0.
#   5. add_derived_column raises clearly if table not found.
#   6. Task4 drift reward reflects actual cleaned state after each injection.
# """
# import os, sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import uuid
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# from models import DataCleanAction, DataCleanObservation, State
# from server.dataset_factory import make_task, generate_drift_batch
# from server.graders import grade_task1, grade_task2, grade_task3, grade_task4

# DRIFT_EVERY = 5

# TASK_CONFIG = {
#     "task1": {
#         "name": "Null Fixer", "difficulty": "easy", "max_steps": 10,
#         "available_ops": ["fill_nulls", "cast_column", "submit"],
#         "description": (
#             "You have a 'main' table (50-row customers dataset). "
#             "Fix: (1) 'age' stored as strings with null markers — fill with median then cast to int. "
#             "(2) 'salary' has ~8 NaN values — fill with mean, keep as float. "
#             "Goal: age is int64 with 0 nulls; salary is float64 with 0 nulls."
#         ),
#     },
#     "task2": {
#         "name": "Schema Normalizer", "difficulty": "medium", "max_steps": 20,
#         "available_ops": ["fill_nulls", "cast_column", "remove_duplicates",
#                           "normalize_values", "submit"],
#         "description": (
#             "You have a 'main' table (~200-row orders). "
#             "Fix: (1) ~30 duplicate rows → remove_duplicates. "
#             "(2) 'country' inconsistent casing → normalize_values(method=upper). "
#             "(3) 'order_date' mixed formats → cast_column(dtype=datetime). "
#             "(4) 'amount' ~12 NaN → fill_nulls(strategy=mean). "
#             "Order: remove_duplicates → normalize_values → cast_column → fill_nulls → submit."
#         ),
#     },
#     "task3": {
#         "name": "ETL Pipeline", "difficulty": "hard", "max_steps": 30,
#         "available_ops": [
#             "fill_nulls", "cast_column", "remove_duplicates", "normalize_values",
#             "filter_outliers", "merge_tables", "add_derived_column", "submit",
#         ],
#         "description": (
#             "Two tables: 'orders' (300 rows) + 'customers' (100 rows). "
#             "Steps: (1) merge_tables(orders,customers,customer_id,merged). "
#             "(2) fill_nulls+cast_column age→int in 'merged'. "
#             "(3) filter_outliers(amount,iqr,1.5,merged). "
#             "(4) add_derived_column(order_year,order_date,year_from_date,merged). "
#             "(5) submit."
#         ),
#     },
#     "task4_data_drift": {
#         "name": "Data Drift (Streaming)", "difficulty": "expert", "max_steps": 40,
#         "available_ops": [
#             "fill_nulls", "cast_column", "remove_duplicates", "normalize_values",
#             "filter_outliers", "submit",
#         ],
#         "description": (
#             "NOVEL TASK — Live streaming transactions under continuous data drift. "
#             f"Starts with 120 dirty rows. Every {DRIFT_EVERY} steps, 7 fresh dirty rows "
#             "are automatically injected — simulating a real Kafka/streaming pipeline. "
#             "Strategy: filter_outliers → fill_nulls → cast_column → submit."
#         ),
#     },
# }


# class DataCleanEnvironment:
#     SUPPORTS_CONCURRENT_SESSIONS = True

#     def __init__(self):
#         self._task_id    = "task1"
#         self._seed       = 42
#         self._episode_id = str(uuid.uuid4())
#         self._step_count = 0
#         self._drift_batch_num   = 0
#         self._tables:            Dict[str, pd.DataFrame] = {}
#         self._expected_tables:   Dict[str, pd.DataFrame] = {}
#         self._dirty_tables:      Dict[str, pd.DataFrame] = {}
#         self._prev_score         = 0.05
#         self._last_reward        = 0.0
#         self._last_msg           = "Not started. Call /reset first."
#         self.last_partial_score  = 0.05

#     def reset(self, task_id: str = "task1", seed: int = 42) -> DataCleanObservation:
#         if task_id not in TASK_CONFIG:
#             task_id = "task1"
#         self._task_id         = task_id
#         self._seed            = seed
#         self._episode_id      = str(uuid.uuid4())
#         self._step_count      = 0
#         self._drift_batch_num = 0
#         self._last_reward     = 0.0

#         dirty, expected = make_task(task_id, seed)
#         self._tables          = {k: v.copy() for k, v in dirty.items()}
#         self._expected_tables = expected
#         self._dirty_tables    = {k: v.copy() for k, v in dirty.items()}

#         initial_score       = self._score()
#         self._prev_score    = initial_score
#         self.last_partial_score = initial_score
#         self._last_msg = (
#             f"Episode started | task={task_id} | seed={seed} | "
#             f"tables={list(self._tables.keys())}"
#         )
#         return self._obs(reward=0.0, done=False, new_rows=0, score=initial_score)

#     def step(self, action: DataCleanAction) -> Tuple[DataCleanObservation, float, bool, dict]:
#         self._step_count += 1
#         max_steps = TASK_CONFIG[self._task_id]["max_steps"]
#         allowed   = TASK_CONFIG[self._task_id]["available_ops"]
#         new_rows_injected = 0

#         # ── Drift injection (task4 only) ──────────────────────────────────────
#         if (self._task_id == "task4_data_drift"
#                 and self._step_count % DRIFT_EVERY == 0):
#             batch = generate_drift_batch(self._seed, self._drift_batch_num, n_rows=7)
#             self._tables["stream"] = pd.concat(
#                 [self._tables["stream"], batch], ignore_index=True
#             )
#             self._drift_batch_num += 1
#             new_rows_injected = len(batch)
#             self._last_msg = (
#                 f"[DRIFT] {new_rows_injected} new dirty rows injected into 'stream' "
#                 f"(batch {self._drift_batch_num}). Keep cleaning!"
#             )

#         # ── Validate operation ────────────────────────────────────────────────
#         if action.operation not in allowed:
#             self._last_msg = (
#                 f"Operation '{action.operation}' not allowed for {self._task_id}. "
#                 f"Allowed: {allowed}"
#             )
#             done = self._step_count >= max_steps
#             reward = -0.02
#             cur_score = self._score()
#             self.last_partial_score = cur_score
#             return self._obs(reward, done, new_rows_injected, score=cur_score), reward, done, {}

#         # ── Submit ────────────────────────────────────────────────────────────
#         if action.operation == "submit":
#             final = self._score()
#             reward = round(final - self._prev_score, 4)
#             self._last_msg = (
#                 f"Submitted! Final score: {final:.4f} | Steps: {self._step_count}/{max_steps}"
#             )
#             self.last_partial_score = final
#             return self._obs(reward, True, new_rows_injected, score=final), reward, True, {}

#         # ── Execute operation ─────────────────────────────────────────────────
#         try:
#             op_msg = self._execute(action)
#             if new_rows_injected == 0:
#                 self._last_msg = op_msg
#             else:
#                 self._last_msg += f" | Action: {op_msg}"
#         except (KeyError, ValueError, TypeError) as exc:
#             self._last_msg = f"Error executing '{action.operation}': {exc}"
#             done = self._step_count >= max_steps
#             reward = -0.02
#             cur_score = self._score()
#             self.last_partial_score = cur_score
#             return self._obs(reward, done, new_rows_injected, score=cur_score), reward, done, {}

#         new_score = self._score()
#         reward    = round(new_score - self._prev_score, 4)
#         self._prev_score        = new_score
#         self.last_partial_score = new_score

#         done = self._step_count >= max_steps
#         if done:
#             self._last_msg += f" | Max steps reached. Score: {new_score:.4f}"

#         return self._obs(reward, done, new_rows_injected, score=new_score), reward, done, {}

#     def state(self) -> State:
#         return State(episode_id=self._episode_id, step_count=self._step_count)

#     # ── Real grader dispatch ──────────────────────────────────────────────────

#     def _score(self) -> float:
#         """Call the appropriate real grader with actual DataFrames."""
#         try:
#             if self._task_id == "task1":
#                 df = self._tables.get("main", pd.DataFrame())
#                 exp = self._expected_tables.get("main", pd.DataFrame())
#                 return grade_task1(df, exp)

#             elif self._task_id == "task2":
#                 df = self._tables.get("main", pd.DataFrame())
#                 exp = self._expected_tables.get("main", pd.DataFrame())
#                 dirty = self._dirty_tables.get("main", pd.DataFrame())
#                 return grade_task2(df, exp, dirty)

#             elif self._task_id == "task3":
#                 exp = self._expected_tables.get("main", pd.DataFrame())
#                 return grade_task3(self._tables, exp, self._dirty_tables)

#             elif self._task_id == "task4_data_drift":
#                 df = self._tables.get("stream", pd.DataFrame())
#                 return grade_task4(df)

#         except Exception:
#             pass

#         return 0.05  # safe floor if grader crashes

#     # ── Operation executor ────────────────────────────────────────────────────

#     def _execute(self, action: DataCleanAction) -> str:
#         op  = action.operation
#         tbl = action.table_name or (
#             "stream" if self._task_id == "task4_data_drift" else "main"
#         )

#         if op == "fill_nulls":
#             df  = self._tbl(tbl)
#             col = self._col(df, action.column, tbl)

#             # Normalise string null markers → real NaN
#             df[col] = df[col].replace(
#                 ["N/A", "n/a", "null", "NULL", "None", "none",
#                  "missing", "", "NaN", "nan"], np.nan
#             )

#             s = (action.strategy or "mean").lower()

#             if s in ("forward_fill", "ffill"):
#                 df[col] = df[col].ffill()
#                 self._tables[tbl] = df
#                 return f"[fill_nulls] '{col}' ffill in '{tbl}'"
#             if s in ("backward_fill", "bfill"):
#                 df[col] = df[col].bfill()
#                 self._tables[tbl] = df
#                 return f"[fill_nulls] '{col}' bfill in '{tbl}'"

#             # Compute fill value on numeric-coerced series for numeric strategies
#             num = pd.to_numeric(df[col], errors="coerce")
#             if s == "mean":
#                 fv = num.mean()
#             elif s == "median":
#                 fv = num.median()
#             elif s == "mode":
#                 # Use numeric mode if column has any numeric data, else raw mode
#                 if num.notna().any():
#                     modes = num.dropna()
#                     fv = modes.mode().iloc[0] if not modes.mode().empty else np.nan
#                 else:
#                     modes = df[col].dropna()
#                     fv = modes.mode().iloc[0] if not modes.mode().empty else np.nan
#             elif s == "constant":
#                 fv = action.value
#             else:
#                 raise ValueError(f"Unknown strategy '{s}'")

#             if fv is None or (isinstance(fv, float) and np.isnan(fv)):
#                 raise ValueError(f"Cannot compute fill value for '{col}' with strategy '{s}'")

#             # Fill NaN in the (already normalised) column using numeric coercion
#             df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fv)
#             self._tables[tbl] = df
#             return f"[fill_nulls] '{col}' → {s}={fv:.4g} in '{tbl}'"

#         elif op == "cast_column":
#             df  = self._tbl(tbl)
#             col = self._col(df, action.column, tbl)
#             dt  = (action.dtype or "").lower()

#             if dt == "int":
#                 num = pd.to_numeric(df[col], errors="coerce")
#                 # Only cast if no nulls remain — otherwise raise informatively
#                 if num.isna().any():
#                     raise ValueError(
#                         f"Column '{col}' still has {int(num.isna().sum())} null(s). "
#                         f"Run fill_nulls first before casting to int."
#                     )
#                 df[col] = num.astype("int64")
#             elif dt == "float":
#                 df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
#             elif dt in ("str", "string", "object"):
#                 df[col] = df[col].astype(str)
#             elif dt == "datetime":
#                 df[col] = pd.to_datetime(df[col], errors="coerce")
#             elif dt == "bool":
#                 df[col] = df[col].astype(bool)
#             else:
#                 raise ValueError(f"Unknown dtype '{dt}'")

#             self._tables[tbl] = df
#             return f"[cast_column] '{col}' → {dt} in '{tbl}'"

#         elif op == "remove_duplicates":
#             df     = self._tbl(tbl)
#             before = len(df)
#             keep   = action.keep if action.keep in ("first", "last") else "first"
#             df     = df.drop_duplicates(subset=action.subset, keep=keep).reset_index(drop=True)
#             self._tables[tbl] = df
#             return f"[remove_duplicates] -{before - len(df)} rows from '{tbl}'"

#         elif op == "normalize_values":
#             df  = self._tbl(tbl)
#             col = self._col(df, action.column, tbl)
#             m   = (action.method or "upper").lower()
#             s   = df[col].astype(str)
#             if m == "upper":      df[col] = s.str.strip().str.upper()
#             elif m == "lower":    df[col] = s.str.strip().str.lower()
#             elif m == "strip":    df[col] = s.str.strip()
#             elif m == "title":    df[col] = s.str.strip().str.title()
#             elif m in ("regex", "replace"):
#                 if not action.pattern:
#                     raise ValueError(f"method='{m}' requires 'pattern'")
#                 df[col] = s.str.replace(
#                     action.pattern, action.replacement or "", regex=(m == "regex")
#                 )
#             else:
#                 raise ValueError(f"Unknown normalize method '{m}'")
#             self._tables[tbl] = df
#             return f"[normalize_values] '{col}' → {m} in '{tbl}'"

#         elif op == "filter_outliers":
#             df  = self._tbl(tbl)
#             col = self._col(df, action.column, tbl)
#             m   = (action.method or "iqr").lower()
#             thr = action.threshold if action.threshold is not None else 1.5
#             num = pd.to_numeric(df[col], errors="coerce")
#             before = len(df)

#             if m == "iqr":
#                 Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
#                 IQR = Q3 - Q1
#                 mask = (num >= Q1 - thr * IQR) & (num <= Q3 + thr * IQR)
#             elif m == "zscore":
#                 z    = (num - num.mean()) / num.std()
#                 mask = z.abs() <= thr
#             else:
#                 raise ValueError(f"Unknown outlier method '{m}'")

#             df = df[mask | num.isna()].reset_index(drop=True)
#             self._tables[tbl] = df
#             return f"[filter_outliers] -{before - len(df)} rows from '{col}' in '{tbl}'"

#         elif op == "merge_tables":
#             lt  = action.left_table  or "orders"
#             rt  = action.right_table or "customers"
#             on  = action.on          or "customer_id"
#             how = action.how if action.how in ("inner", "left", "right", "outer") else "inner"
#             out = action.output_table or "merged"
#             L, R = self._tbl(lt), self._tbl(rt)
#             if on not in L.columns:
#                 raise ValueError(f"Key '{on}' not in '{lt}'. Available: {list(L.columns)}")
#             if on not in R.columns:
#                 raise ValueError(f"Key '{on}' not in '{rt}'. Available: {list(R.columns)}")
#             merged = pd.merge(L, R, on=on, how=how)
#             self._tables[out] = merged
#             return f"[merge_tables] '{lt}'×'{rt}' on '{on}' → '{out}' ({len(merged)} rows)"

#         elif op == "add_derived_column":
#             # Prefer action.table_name, then try "merged", then "main"
#             tbl2 = action.table_name
#             if tbl2 and tbl2 in self._tables:
#                 pass  # use as-is
#             elif "merged" in self._tables:
#                 tbl2 = "merged"
#             elif "main" in self._tables:
#                 tbl2 = "main"
#             else:
#                 raise KeyError(
#                     f"Table '{tbl2}' not found and no 'merged' or 'main' fallback. "
#                     f"Run merge_tables first. Available: {list(self._tables.keys())}"
#                 )

#             df  = self._tbl(tbl2)
#             cn  = action.column_name
#             src = action.source_column
#             tr  = (action.transform or "").lower()

#             if not cn:  raise ValueError("'column_name' is required")
#             if not src: raise ValueError("'source_column' is required")
#             self._col(df, src, tbl2)  # validates column exists

#             if tr == "year_from_date":
#                 df[cn] = pd.to_datetime(df[src], errors="coerce").dt.year
#             elif tr == "month_from_date":
#                 df[cn] = pd.to_datetime(df[src], errors="coerce").dt.month
#             elif tr == "log1p":
#                 df[cn] = np.log1p(pd.to_numeric(df[src], errors="coerce"))
#             elif tr == "abs":
#                 df[cn] = pd.to_numeric(df[src], errors="coerce").abs()
#             elif tr == "len":
#                 df[cn] = df[src].astype(str).str.len()
#             elif tr in ("upper", "lower"):
#                 df[cn] = getattr(df[src].astype(str).str, tr)()
#             else:
#                 raise ValueError(f"Unknown transform '{tr}'")

#             self._tables[tbl2] = df
#             return f"[add_derived_column] '{cn}'={tr}('{src}') in '{tbl2}'"

#         raise ValueError(f"Unknown operation '{op}'")

#     # ── Observation builder ───────────────────────────────────────────────────

#     def _obs(self, reward: float, done: bool,
#              new_rows: int = 0, score: Optional[float] = None) -> DataCleanObservation:
#         cfg   = TASK_CONFIG[self._task_id]
#         score = score if score is not None else self._score()

#         tables_json, col_dtypes, null_counts, dup_counts, row_counts = {}, {}, {}, {}, {}
#         for nm, df in self._tables.items():
#             tables_json[nm] = df.head(10).to_json(orient="records", default_handler=str)
#             col_dtypes[nm]  = {c: str(df[c].dtype) for c in df.columns}
#             null_counts[nm] = {c: int(df[c].isna().sum()) for c in df.columns}
#             dup_counts[nm]  = int(df.duplicated().sum())
#             row_counts[nm]  = int(len(df))

#         msg = self._last_msg
#         if new_rows > 0 and "[DRIFT]" not in msg:
#             msg = f"[+{new_rows} drift rows] " + msg

#         return DataCleanObservation(
#             task_id=self._task_id,
#             task_description=cfg["description"],
#             step_count=self._step_count,
#             max_steps=cfg["max_steps"],
#             message=msg,
#             tables=tables_json,
#             column_dtypes=col_dtypes,
#             null_counts=null_counts,
#             duplicate_count=dup_counts,
#             row_count=row_counts,
#             schema_errors=self._schema_errors(),
#             available_operations=cfg["available_ops"],
#             reward=round(float(reward), 4),
#             done=done,
#             partial_score=round(float(score), 4),
#         )

#     def _schema_errors(self) -> List[str]:
#         errs = []
#         for nm, df in self._tables.items():
#             for col in df.columns:
#                 nc = int(df[col].isna().sum())
#                 if nc:
#                     errs.append(f"{nm}.{col}: {nc} nulls")
#             dc = int(df.duplicated().sum())
#             if dc:
#                 errs.append(f"{nm}: {dc} duplicates")
#         return errs[:12]

#     def _tbl(self, name: str) -> pd.DataFrame:
#         if name not in self._tables:
#             raise KeyError(
#                 f"Table '{name}' not found. Available: {list(self._tables.keys())}"
#             )
#         return self._tables[name]

#     @staticmethod
#     def _col(df: pd.DataFrame, col: Optional[str], tbl: str) -> str:
#         if not col:
#             raise ValueError("'column' field is required")
#         if col not in df.columns:
#             raise KeyError(
#                 f"Column '{col}' not in '{tbl}'. Available: {list(df.columns)}"
#             )
#         return col

"""
DataClean Environment — core logic.
All bugs fixed:
  1. _score() now calls real graders with actual DataFrames.
  2. Reward delta is real: new_score - prev_score every step.
  3. fill_nulls mode strategy uses numeric mode, not raw-string mode.
  4. cast_column int no longer silently fills NaN with 0.
  5. add_derived_column raises clearly if table not found.
  6. Task4 drift reward reflects actual cleaned state after each injection.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import DataCleanAction, DataCleanObservation, State
from server.dataset_factory import make_task, generate_drift_batch
from server.graders import grade_task1, grade_task2, grade_task3, grade_task4

DRIFT_EVERY = 5

TASK_CONFIG = {
    "task1": {
        "name": "Null Fixer", "difficulty": "easy", "max_steps": 10,
        "available_ops": ["fill_nulls", "cast_column", "submit"],
        "description": (
            "You have a 'main' table (50-row customers dataset). "
            "Fix: (1) 'age' stored as strings with null markers — fill with median then cast to int. "
            "(2) 'salary' has ~8 NaN values — fill with mean, keep as float. "
            "Goal: age is int64 with 0 nulls; salary is float64 with 0 nulls."
        ),
    },
    "task2": {
        "name": "Schema Normalizer", "difficulty": "medium", "max_steps": 20,
        "available_ops": ["fill_nulls", "cast_column", "remove_duplicates",
                          "normalize_values", "submit"],
        "description": (
            "You have a 'main' table (~200-row orders). "
            "Fix: (1) ~30 duplicate rows → remove_duplicates. "
            "(2) 'country' inconsistent casing → normalize_values(method=upper). "
            "(3) 'order_date' mixed formats → cast_column(dtype=datetime). "
            "(4) 'amount' ~12 NaN → fill_nulls(strategy=mean). "
            "Order: remove_duplicates → normalize_values → cast_column → fill_nulls → submit."
        ),
    },
    "task3": {
        "name": "ETL Pipeline", "difficulty": "hard", "max_steps": 30,
        "available_ops": [
            "fill_nulls", "cast_column", "remove_duplicates", "normalize_values",
            "filter_outliers", "merge_tables", "add_derived_column", "submit",
        ],
        "description": (
            "Two tables: 'orders' (300 rows) + 'customers' (100 rows). "
            "Steps: (1) merge_tables(orders,customers,customer_id,merged). "
            "(2) fill_nulls+cast_column age→int in 'merged'. "
            "(3) filter_outliers(amount,iqr,1.5,merged). "
            "(4) add_derived_column(order_year,order_date,year_from_date,merged). "
            "(5) submit."
        ),
    },
    "task4_data_drift": {
        "name": "Data Drift (Streaming)", "difficulty": "expert", "max_steps": 40,
        "available_ops": [
            "fill_nulls", "cast_column", "remove_duplicates", "normalize_values",
            "filter_outliers", "submit",
        ],
        "description": (
            "NOVEL TASK — Live streaming transactions under continuous data drift. "
            f"Starts with 120 dirty rows. Every {DRIFT_EVERY} steps, 7 fresh dirty rows "
            "are automatically injected — simulating a real Kafka/streaming pipeline. "
            "Strategy: filter_outliers → fill_nulls → cast_column → submit."
        ),
    },
}


class DataCleanEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._task_id    = "task1"
        self._seed       = 42
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._drift_batch_num   = 0
        self._tables:            Dict[str, pd.DataFrame] = {}
        self._expected_tables:   Dict[str, pd.DataFrame] = {}
        self._dirty_tables:      Dict[str, pd.DataFrame] = {}
        self._prev_score         = 0.05
        self._last_reward        = 0.0
        self._last_msg           = "Not started. Call /reset first."
        self.last_partial_score  = 0.05

    def reset(self, task_id: str = "task1", seed: int = 42) -> DataCleanObservation:
        if task_id not in TASK_CONFIG:
            task_id = "task1"
        self._task_id         = task_id
        self._seed            = seed
        self._episode_id      = str(uuid.uuid4())
        self._step_count      = 0
        self._drift_batch_num = 0
        self._last_reward     = 0.0

        dirty, expected = make_task(task_id, seed)
        self._tables          = {k: v.copy() for k, v in dirty.items()}
        self._expected_tables = expected
        self._dirty_tables    = {k: v.copy() for k, v in dirty.items()}

        initial_score       = self._score()
        self._prev_score    = initial_score
        self.last_partial_score = initial_score
        self._last_msg = (
            f"Episode started | task={task_id} | seed={seed} | "
            f"tables={list(self._tables.keys())}"
        )
        return self._obs(reward=0.0, done=False, new_rows=0, score=initial_score)

    def step(self, action: DataCleanAction) -> Tuple[DataCleanObservation, float, bool, dict]:
        self._step_count += 1
        max_steps = TASK_CONFIG[self._task_id]["max_steps"]
        allowed   = TASK_CONFIG[self._task_id]["available_ops"]
        new_rows_injected = 0

        # ── Drift injection (task4 only) ──────────────────────────────────────
        if (self._task_id == "task4_data_drift"
                and self._step_count % DRIFT_EVERY == 0):
            batch = generate_drift_batch(self._seed, self._drift_batch_num, n_rows=7)
            self._tables["stream"] = pd.concat(
                [self._tables["stream"], batch], ignore_index=True
            )
            self._drift_batch_num += 1
            new_rows_injected = len(batch)
            self._last_msg = (
                f"[DRIFT] {new_rows_injected} new dirty rows injected into 'stream' "
                f"(batch {self._drift_batch_num}). Keep cleaning!"
            )

        # ── Validate operation ────────────────────────────────────────────────
        if action.operation not in allowed:
            self._last_msg = (
                f"Operation '{action.operation}' not allowed for {self._task_id}. "
                f"Allowed: {allowed}"
            )
            done = self._step_count >= max_steps
            reward = -0.02
            cur_score = self._score()
            self.last_partial_score = cur_score
            return self._obs(reward, done, new_rows_injected, score=cur_score), reward, done, {}

        # ── Submit ────────────────────────────────────────────────────────────
        if action.operation == "submit":
            final = self._score()
            reward = round(final - self._prev_score, 4)
            self._last_msg = (
                f"Submitted! Final score: {final:.4f} | Steps: {self._step_count}/{max_steps}"
            )
            self.last_partial_score = final
            return self._obs(reward, True, new_rows_injected, score=final), reward, True, {}

        # ── Execute operation ─────────────────────────────────────────────────
        try:
            op_msg = self._execute(action)
            if new_rows_injected == 0:
                self._last_msg = op_msg
            else:
                self._last_msg += f" | Action: {op_msg}"
        except (KeyError, ValueError, TypeError) as exc:
            self._last_msg = f"Error executing '{action.operation}': {exc}"
            done = self._step_count >= max_steps
            reward = -0.02
            cur_score = self._score()
            self.last_partial_score = cur_score
            return self._obs(reward, done, new_rows_injected, score=cur_score), reward, done, {}

        new_score = self._score()
        reward    = round(new_score - self._prev_score, 4)
        self._prev_score        = new_score
        self.last_partial_score = new_score

        done = self._step_count >= max_steps
        if done:
            self._last_msg += f" | Max steps reached. Score: {new_score:.4f}"

        return self._obs(reward, done, new_rows_injected, score=new_score), reward, done, {}

    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._step_count)

    # ── Real grader dispatch ──────────────────────────────────────────────────

    def _score(self) -> float:
        """Call the appropriate real grader with actual DataFrames."""
        try:
            if self._task_id == "task1":
                df = self._tables.get("main", pd.DataFrame())
                exp = self._expected_tables.get("main", pd.DataFrame())
                return grade_task1(df, exp)

            elif self._task_id == "task2":
                df = self._tables.get("main", pd.DataFrame())
                exp = self._expected_tables.get("main", pd.DataFrame())
                dirty = self._dirty_tables.get("main", pd.DataFrame())
                return grade_task2(df, exp, dirty)

            elif self._task_id == "task3":
                exp = self._expected_tables.get("main", pd.DataFrame())
                return grade_task3(self._tables, exp, self._dirty_tables)

            elif self._task_id == "task4_data_drift":
                df = self._tables.get("stream", pd.DataFrame())
                return grade_task4(df)

        except Exception:
            pass

        return 0.05  # safe floor if grader crashes

    # ── Operation executor ────────────────────────────────────────────────────

    def _execute(self, action: DataCleanAction) -> str:
        op  = action.operation
        tbl = action.table_name or (
            "stream" if self._task_id == "task4_data_drift" else "main"
        )

        if op == "fill_nulls":
            df  = self._tbl(tbl)
            col = self._col(df, action.column, tbl)

            # Normalise string null markers → real NaN
            df[col] = df[col].replace(
                ["N/A", "n/a", "null", "NULL", "None", "none",
                 "missing", "", "NaN", "nan"], np.nan
            )

            s = (action.strategy or "mean").lower()

            if s in ("forward_fill", "ffill"):
                df[col] = df[col].ffill()
                self._tables[tbl] = df
                return f"[fill_nulls] '{col}' ffill in '{tbl}'"
            if s in ("backward_fill", "bfill"):
                df[col] = df[col].bfill()
                self._tables[tbl] = df
                return f"[fill_nulls] '{col}' bfill in '{tbl}'"

            # Determine if column is primarily numeric or string
            num = pd.to_numeric(df[col], errors="coerce")
            is_numeric_col = num.notna().sum() > df[col].notna().sum() * 0.5

            if s == "mean":
                fv = num.mean()
            elif s == "median":
                fv = num.median()
            elif s == "mode":
                if is_numeric_col:
                    # Numeric column: use mode of numeric values
                    valid = num.dropna()
                    fv = valid.mode().iloc[0] if not valid.mode().empty else np.nan
                else:
                    # String/categorical column: use raw string mode, then fill directly
                    raw_valid = df[col].dropna().astype(str)
                    raw_valid = raw_valid[raw_valid.str.lower() != "nan"]
                    if raw_valid.empty:
                        raise ValueError(f"Cannot compute mode for '{col}' — all values are null")
                    fv_str = str(raw_valid.mode().iloc[0])
                    df[col] = df[col].fillna(fv_str)
                    self._tables[tbl] = df
                    return f"[fill_nulls] '{col}' → mode='{fv_str}' in '{tbl}'"
            elif s == "constant":
                fv = action.value
            else:
                raise ValueError(f"Unknown strategy '{s}'")

            if fv is None or (isinstance(fv, float) and np.isnan(fv)):
                raise ValueError(f"Cannot compute fill value for '{col}' with strategy '{s}'")

            if is_numeric_col:
                # Numeric column: coerce to numeric and fill
                df[col] = num.fillna(fv)
            else:
                # String column with numeric fill (e.g. constant=0) — fill as-is
                df[col] = df[col].fillna(fv)
            self._tables[tbl] = df
            fv_display = f"{fv:.4g}" if isinstance(fv, (int, float)) else str(fv)
            return f"[fill_nulls] '{col}' → {s}={fv_display} in '{tbl}'"

        elif op == "cast_column":
            df  = self._tbl(tbl)
            col = self._col(df, action.column, tbl)
            dt  = (action.dtype or "").lower()

            if dt == "int":
                num = pd.to_numeric(df[col], errors="coerce")
                # Only cast if no nulls remain — otherwise raise informatively
                if num.isna().any():
                    raise ValueError(
                        f"Column '{col}' still has {int(num.isna().sum())} null(s). "
                        f"Run fill_nulls first before casting to int."
                    )
                df[col] = num.astype("int64")
            elif dt == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dt in ("str", "string", "object"):
                df[col] = df[col].astype(str)
            elif dt == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dt == "bool":
                df[col] = df[col].astype(bool)
            else:
                raise ValueError(f"Unknown dtype '{dt}'")

            self._tables[tbl] = df
            return f"[cast_column] '{col}' → {dt} in '{tbl}'"

        elif op == "remove_duplicates":
            df     = self._tbl(tbl)
            before = len(df)
            keep   = action.keep if action.keep in ("first", "last") else "first"
            df     = df.drop_duplicates(subset=action.subset, keep=keep).reset_index(drop=True)
            self._tables[tbl] = df
            return f"[remove_duplicates] -{before - len(df)} rows from '{tbl}'"

        elif op == "normalize_values":
            df  = self._tbl(tbl)
            col = self._col(df, action.column, tbl)
            m   = (action.method or "upper").lower()
            s   = df[col].astype(str)
            if m == "upper":      df[col] = s.str.strip().str.upper()
            elif m == "lower":    df[col] = s.str.strip().str.lower()
            elif m == "strip":    df[col] = s.str.strip()
            elif m == "title":    df[col] = s.str.strip().str.title()
            elif m in ("regex", "replace"):
                if not action.pattern:
                    raise ValueError(f"method='{m}' requires 'pattern'")
                df[col] = s.str.replace(
                    action.pattern, action.replacement or "", regex=(m == "regex")
                )
            else:
                raise ValueError(f"Unknown normalize method '{m}'")
            self._tables[tbl] = df
            return f"[normalize_values] '{col}' → {m} in '{tbl}'"

        elif op == "filter_outliers":
            df  = self._tbl(tbl)
            col = self._col(df, action.column, tbl)
            m   = (action.method or "iqr").lower()
            thr = action.threshold if action.threshold is not None else 1.5
            num = pd.to_numeric(df[col], errors="coerce")
            before = len(df)

            if m == "iqr":
                Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
                IQR = Q3 - Q1
                mask = (num >= Q1 - thr * IQR) & (num <= Q3 + thr * IQR)
            elif m == "zscore":
                z    = (num - num.mean()) / num.std()
                mask = z.abs() <= thr
            else:
                raise ValueError(f"Unknown outlier method '{m}'")

            df = df[mask | num.isna()].reset_index(drop=True)
            self._tables[tbl] = df
            return f"[filter_outliers] -{before - len(df)} rows from '{col}' in '{tbl}'"

        elif op == "merge_tables":
            lt  = action.left_table  or "orders"
            rt  = action.right_table or "customers"
            on  = action.on          or "customer_id"
            how = action.how if action.how in ("inner", "left", "right", "outer") else "inner"
            out = action.output_table or "merged"
            L, R = self._tbl(lt), self._tbl(rt)
            if on not in L.columns:
                raise ValueError(f"Key '{on}' not in '{lt}'. Available: {list(L.columns)}")
            if on not in R.columns:
                raise ValueError(f"Key '{on}' not in '{rt}'. Available: {list(R.columns)}")
            merged = pd.merge(L, R, on=on, how=how)
            self._tables[out] = merged
            return f"[merge_tables] '{lt}'×'{rt}' on '{on}' → '{out}' ({len(merged)} rows)"

        elif op == "add_derived_column":
            # Prefer action.table_name, then try "merged", then "main"
            tbl2 = action.table_name
            if tbl2 and tbl2 in self._tables:
                pass  # use as-is
            elif "merged" in self._tables:
                tbl2 = "merged"
            elif "main" in self._tables:
                tbl2 = "main"
            else:
                raise KeyError(
                    f"Table '{tbl2}' not found and no 'merged' or 'main' fallback. "
                    f"Run merge_tables first. Available: {list(self._tables.keys())}"
                )

            df  = self._tbl(tbl2)
            cn  = action.column_name
            src = action.source_column
            tr  = (action.transform or "").lower()

            if not cn:  raise ValueError("'column_name' is required")
            if not src: raise ValueError("'source_column' is required")
            self._col(df, src, tbl2)  # validates column exists

            if tr == "year_from_date":
                df[cn] = pd.to_datetime(df[src], errors="coerce").dt.year
            elif tr == "month_from_date":
                df[cn] = pd.to_datetime(df[src], errors="coerce").dt.month
            elif tr == "log1p":
                df[cn] = np.log1p(pd.to_numeric(df[src], errors="coerce"))
            elif tr == "abs":
                df[cn] = pd.to_numeric(df[src], errors="coerce").abs()
            elif tr == "len":
                df[cn] = df[src].astype(str).str.len()
            elif tr in ("upper", "lower"):
                df[cn] = getattr(df[src].astype(str).str, tr)()
            else:
                raise ValueError(f"Unknown transform '{tr}'")

            self._tables[tbl2] = df
            return f"[add_derived_column] '{cn}'={tr}('{src}') in '{tbl2}'"

        raise ValueError(f"Unknown operation '{op}'")

    # ── Observation builder ───────────────────────────────────────────────────

    def _obs(self, reward: float, done: bool,
             new_rows: int = 0, score: Optional[float] = None) -> DataCleanObservation:
        cfg   = TASK_CONFIG[self._task_id]
        score = score if score is not None else self._score()

        tables_json, col_dtypes, null_counts, dup_counts, row_counts = {}, {}, {}, {}, {}
        for nm, df in self._tables.items():
            tables_json[nm] = df.head(10).to_json(orient="records", default_handler=str)
            col_dtypes[nm]  = {c: str(df[c].dtype) for c in df.columns}
            null_counts[nm] = {c: int(df[c].isna().sum()) for c in df.columns}
            dup_counts[nm]  = int(df.duplicated().sum())
            row_counts[nm]  = int(len(df))

        msg = self._last_msg
        if new_rows > 0 and "[DRIFT]" not in msg:
            msg = f"[+{new_rows} drift rows] " + msg

        return DataCleanObservation(
            task_id=self._task_id,
            task_description=cfg["description"],
            step_count=self._step_count,
            max_steps=cfg["max_steps"],
            message=msg,
            tables=tables_json,
            column_dtypes=col_dtypes,
            null_counts=null_counts,
            duplicate_count=dup_counts,
            row_count=row_counts,
            schema_errors=self._schema_errors(),
            available_operations=cfg["available_ops"],
            reward=round(float(reward), 4),
            done=done,
            partial_score=round(float(score), 4),
        )

    def _schema_errors(self) -> List[str]:
        errs = []
        for nm, df in self._tables.items():
            for col in df.columns:
                nc = int(df[col].isna().sum())
                if nc:
                    errs.append(f"{nm}.{col}: {nc} nulls")
            dc = int(df.duplicated().sum())
            if dc:
                errs.append(f"{nm}: {dc} duplicates")
        return errs[:12]

    def _tbl(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(
                f"Table '{name}' not found. Available: {list(self._tables.keys())}"
            )
        return self._tables[name]

    @staticmethod
    def _col(df: pd.DataFrame, col: Optional[str], tbl: str) -> str:
        if not col:
            raise ValueError("'column' field is required")
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not in '{tbl}'. Available: {list(df.columns)}"
            )
        return col