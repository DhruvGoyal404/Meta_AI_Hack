---
title: Dataclean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - data-cleaning
  - etl
  - real-world
  - tabular
  - data-drift
  - streaming
---

# DataClean OpenEnv

A **real-world data cleaning and ETL environment** for AI agent RL training, built on the [OpenEnv](https://meta-pytorch.org/OpenEnv/) framework by Meta-PyTorch and Hugging Face.

Agents learn to fix messy tabular data — filling nulls, normalising inconsistent values, removing duplicates, filtering outliers, performing multi-table ETL joins, and handling **live data drift** (novel: fresh dirty rows injected mid-episode every 5 steps).

---

## Why This Environment?

Data cleaning is one of the most common, time-consuming tasks in real data engineering. Every company with a data pipeline does this daily. Training an RL agent on this task has immediate real-world value — unlike game-based environments.

**What makes this different from other OpenEnv submissions:**
- Task 4 (Data Drift) is genuinely novel — no existing OpenEnv environment simulates live streaming row injection mid-episode
- Parallel baseline using `concurrent.futures.ThreadPoolExecutor` — all 4 tasks run simultaneously
- Deterministic graders using seeded dataset generation — perfectly reproducible scores
- Dense reward signal: reward = grader(new_state) − grader(prev_state) at every step

---

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `task1` | Easy | Fix nulls + dtypes in 50-row customer CSV | 10 |
| `task2` | Medium | Dedup + normalize strings + fix dates + fill nulls | 20 |
| `task3` | Hard | Multi-table merge + outlier removal + derived column | 30 |
| `task4_data_drift` | **Expert** | Live streaming table — 7 dirty rows injected every 5 steps | 40 |

---

## Action Space

Every action is a JSON object with an `operation` field:

```json
{"operation": "fill_nulls",   "column": "age",          "strategy": "median"}
{"operation": "cast_column",  "column": "age",           "dtype": "int"}
{"operation": "remove_duplicates"}
{"operation": "normalize_values", "column": "country",   "method": "upper"}
{"operation": "cast_column",  "column": "order_date",    "dtype": "datetime"}
{"operation": "filter_outliers", "column": "amount",     "method": "iqr", "threshold": 1.5}
{"operation": "merge_tables", "left_table": "orders",    "right_table": "customers", "on": "customer_id", "output_table": "merged"}
{"operation": "add_derived_column", "column_name": "order_year", "source_column": "order_date", "transform": "year_from_date", "table_name": "merged"}
{"operation": "submit"}
```

## Observation Space

After each `reset()` / `step()` the agent receives:

- `task_id`, `task_description`, `step_count`, `max_steps`, `message`
- `tables` — dict of `{table_name → JSON string of df.head(10)}`
- `column_dtypes`, `null_counts`, `duplicate_count`, `row_count`
- `schema_errors` — list of detected problems to guide the agent
- `reward`, `done`, `partial_score` — RL signals

---

## Reward Function

```
step_reward = grader(current_state) − grader(previous_state)   # dense delta signal
invalid_op  = −0.02                                              # bad operation penalty
terminal    = final grader score on submit or max_steps
```

Partial credit per sub-dimension. Score range: `[0.0, 1.0]`.

---

## Baseline Scores (llama-3.3-70b-versatile, seed=42, parallel run)

| Task | Score | Time |
|------|-------|------|
| task1 (easy) | 1.0000 | 11.8s |
| task2 (medium) | 1.0000 | 34.1s |
| task3 (hard) | 0.8000 | 22.4s |
| task4_data_drift (expert) | 0.9297 | 26.6s |
| **mean** | **0.9324** | **34.3s wall** |

All 4 tasks run in parallel — wall time = slowest task, not sum.

---

## Setup & Local Run

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export BASELINE_MODEL=llama-3.3-70b-versatile
python baseline.py
```

## Docker

```bash
docker build -t dataclean-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key dataclean-env
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute one cleaning operation |
| GET | `/state` | Current episode metadata |
| GET | `/tasks` | All tasks + action schema |
| GET | `/grader` | Score current episode state |
| GET | `/baseline` | Run baseline agent on all tasks |
| GET | `/health` | Liveness probe |
| GET | `/docs` | Interactive Swagger UI |

---

## Project Structure

```
dataCleaningProject/
├── server/
│   ├── app.py              # FastAPI server — all endpoints
│   ├── environment.py      # Core env logic — reset/step/state + drift injection
│   ├── graders.py          # Deterministic scoring for all 4 tasks
│   └── dataset_factory.py  # Seeded dirty+expected dataset generation + drift batches
├── models.py               # Pydantic Action + Observation models
├── baseline.py             # Parallel baseline (ThreadPoolExecutor, 4 tasks at once)
├── client.py               # HTTP client
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Port 7860, Python 3.11-slim
└── requirements.txt
```
