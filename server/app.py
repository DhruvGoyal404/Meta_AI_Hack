"""
FastAPI server for DataClean OpenEnv.

FIXES vs original:
  1. /reset now takes a JSON body (ResetRequest) — was query params, baseline.py sent JSON body
  2. All imports are absolute + sys.path patched — was relative (broke with uvicorn from root)
  3. /baseline calls internal agent logic — was importing baseline.py which made HTTP calls (circular)
  4. /step response format: {"observation":..., "reward":..., "done":..., "info":{}}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import logging
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import DataCleanAction, DataCleanObservation, State
from server.environment import DataCleanEnvironment, TASK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataclean")

app = FastAPI(title="DataClean OpenEnv", version="1.0.0",
              description="Real-world data cleaning RL environment — 3 tasks, deterministic graders.")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Session store (multi-agent support) ──────────────────────────────────────
_sessions: Dict[str, DataCleanEnvironment] = {}

def _env(session_id: str = "default") -> DataCleanEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = DataCleanEnvironment()
    return _sessions[session_id]

# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:    str = "task1"
    seed:       int = 42
    session_id: str = "default"


# ── Standard OpenEnv endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "tasks": list(TASK_CONFIG.keys())}

@app.get("/")
def root():
    return {"name": "DataClean OpenEnv", "docs": "/docs",
            "endpoints": ["/reset","/step","/state","/tasks","/grader","/baseline","/health"]}

@app.post("/reset")
def reset(body: Optional[ResetRequest] = Body(default=None)):
    """
    Accepts an optional JSON body — defaults to task1/seed=42/session=default.
    baseline.py sends: requests.post('/reset', json={"task_id":..., "seed":...})
    OpenEnv validator may send POST /reset with no body at all.
    """
    if body is None:
        body = ResetRequest()
    env = _env(body.session_id)
    try:
        obs = env.reset(task_id=body.task_id, seed=body.seed)
        logger.info("reset | session=%s task=%s seed=%d", body.session_id, body.task_id, body.seed)
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/step")
def step(action: DataCleanAction, session_id: str = "default"):
    """Execute one cleaning operation. Returns {observation, reward, done, info}."""
    env = _env(session_id)
    try:
        obs, reward, done, info = env.step(action)
        logger.info("step | session=%s op=%s score=%.4f done=%s",
                    session_id, action.operation, obs.partial_score, done)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/state")
def state(session_id: str = "default"):
    env = _env(session_id)
    s   = env.state()
    return {"episode_id": s.episode_id, "step_count": s.step_count,
            "task_id": env._task_id, "session_id": session_id}

# ── Required hackathon endpoints ──────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """Lists all tasks + action schema. Validators enumerate tasks from here."""
    return {
        "tasks": [
            {
                "id":   tid,
                "name": cfg["name"],
                "difficulty":   cfg["difficulty"],
                "description":  cfg["description"],
                "max_steps":    cfg["max_steps"],
                "available_operations": cfg["available_ops"],
                "action_schema": DataCleanAction.model_json_schema(),
            }
            for tid, cfg in TASK_CONFIG.items()
        ]
    }

@app.get("/grader")
def grader(session_id: str = "default"):
    """Returns current grader score in [0.0, 1.0] for active episode."""
    env = _env(session_id)
    return {"score": env.last_partial_score, "task_id": env._task_id,
            "step_count": env._step_count, "session_id": session_id}

@app.get("/baseline")
async def baseline():
    """
    FIX: Calls internal agent logic — does NOT import baseline.py (was circular).
    Runs GPT-4o-mini (or Groq llama) against all 3 tasks. Requires OPENAI_API_KEY env var.
    """
    try:
        result = await _run_baseline_internal()
        return result
    except Exception as exc:
        logger.error("baseline error: %s", exc)
        return {"error": str(exc), "scores": {}, "mean_score": 0.0}

# ── Internal baseline logic ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent. Respond ONLY with a valid JSON object — no prose, no markdown.

Operations and their JSON fields:
  fill_nulls:         {"operation":"fill_nulls","column":"<col>","strategy":"mean|median|mode|constant|forward_fill|backward_fill"}
  cast_column:        {"operation":"cast_column","column":"<col>","dtype":"int|float|str|datetime"}
  remove_duplicates:  {"operation":"remove_duplicates"}
  normalize_values:   {"operation":"normalize_values","column":"<col>","method":"upper|lower|regex"}
  filter_outliers:    {"operation":"filter_outliers","column":"<col>","method":"iqr|zscore","threshold":1.5,"table_name":"merged"}
  merge_tables:       {"operation":"merge_tables","left_table":"orders","right_table":"customers","on":"customer_id","output_table":"merged"}
  add_derived_column: {"operation":"add_derived_column","column_name":"order_year","source_column":"order_date","transform":"year_from_date","table_name":"merged"}
  submit:             {"operation":"submit"}

Task strategies:
  task1: fill_nulls(age,median)→cast_column(age,int)→fill_nulls(salary,mean)→submit
  task2: remove_duplicates→normalize_values(country,upper)→cast_column(order_date,datetime)→fill_nulls(amount,mean)→submit
  task3: merge_tables→fill_nulls(age)→cast_column(age,int)→filter_outliers(amount,iqr)→add_derived_column(order_year)→submit
"""

async def _run_baseline_internal() -> dict:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {"error": "openai not installed", "scores": {}, "mean_score": 0.0}

    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model    = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")

    if not api_key:
        return {"error": "OPENAI_API_KEY not set", "scores": {}, "mean_score": 0.0}

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    scores: Dict[str, float] = {}

    for task_id, cfg in TASK_CONFIG.items():
        session_id = f"baseline_{task_id}"
        env = _env(session_id)
        obs = env.reset(task_id=task_id, seed=42)

        for _ in range(cfg["max_steps"]):
            if obs.done:
                break
            obs_d  = obs.model_dump()
            prompt = (
                f"Task: {obs_d['task_id']}\nDescription: {obs_d['task_description']}\n"
                f"Step: {obs_d['step_count']}/{obs_d['max_steps']}\n"
                f"Score: {obs_d['partial_score']}\nLast message: {obs_d['message']}\n"
                f"Schema errors: {obs_d['schema_errors'][:5]}\n"
                f"Column dtypes: {json.dumps(obs_d['column_dtypes'])}\n"
                f"Null counts: {json.dumps(obs_d['null_counts'])}\n"
                f"Duplicate counts: {obs_d['duplicate_count']}\n"
                f"Available ops: {obs_d['available_operations']}\n\nNext action JSON:"
            )
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},
                               {"role":"user","content":prompt}],
                    response_format={"type":"json_object"},
                    temperature=0.0, max_tokens=256,
                )
                action = DataCleanAction(**json.loads(resp.choices[0].message.content))
            except Exception:
                action = DataCleanAction(operation="submit")

            obs_tuple = env.step(action)
            obs = obs_tuple[0]

        scores[task_id] = round(float(obs.partial_score), 4)
        logger.info("baseline | task=%s score=%.4f", task_id, scores[task_id])

    mean = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    return {"scores": scores, "mean_score": mean, "model": model, "seed": 42}