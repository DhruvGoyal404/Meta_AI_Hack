"""
DataClean OpenEnv — Inference Script
=====================================
Runs all 4 tasks in parallel using concurrent.futures.ThreadPoolExecutor.
Uses env vars: HF_TOKEN, API_BASE_URL, MODEL_NAME (per hackathon spec).
"""
import os, json, time, sys
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

ENV_URL  = os.environ.get("ENV_URL", "http://localhost:7860")
API_KEY  = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_MAX_STEPS = {
    "task1": 10,
    "task2": 20,
    "task3": 30,
    "task4_data_drift": 40,
}

SYSTEM_PROMPT = """You are an expert data cleaning agent. Respond ONLY with valid JSON — no prose, no markdown.

Operations:
  fill_nulls:         {"operation":"fill_nulls","column":"<col>","strategy":"mean|median|mode|constant","table_name":"<tbl>"}
  cast_column:        {"operation":"cast_column","column":"<col>","dtype":"int|float|str|datetime","table_name":"<tbl>"}
  remove_duplicates:  {"operation":"remove_duplicates","table_name":"<tbl>"}
  normalize_values:   {"operation":"normalize_values","column":"<col>","method":"upper|lower|regex","table_name":"<tbl>"}
  filter_outliers:    {"operation":"filter_outliers","column":"<col>","method":"iqr|zscore","threshold":1.5,"table_name":"<tbl>"}
  merge_tables:       {"operation":"merge_tables","left_table":"orders","right_table":"customers","on":"customer_id","output_table":"merged"}
  add_derived_column: {"operation":"add_derived_column","column_name":"order_year","source_column":"order_date","transform":"year_from_date","table_name":"merged"}
  submit:             {"operation":"submit"}

Task strategies:
  task1: fill_nulls(age,median,main)->cast_column(age,int,main)->fill_nulls(salary,mean,main)->submit
  task2: remove_duplicates(main)->normalize_values(country,upper,main)->cast_column(order_date,datetime,main)->fill_nulls(amount,mean,main)->submit
  task3: merge_tables(orders,customers,customer_id)->fill_nulls(age,median,merged)->cast_column(age,int,merged)->filter_outliers(amount,iqr,1.5,merged)->add_derived_column(order_year,order_date,year_from_date,merged)->submit
  task4_data_drift: filter_outliers(amount,iqr,1.5,stream)->fill_nulls(amount,mean,stream)->cast_column(amount,float,stream)->fill_nulls(category,mode,stream)->fill_nulls(region,mode,stream)->cast_column(event_ts,datetime,stream)->submit
"""


def _make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _build_prompt(obs: dict, task_id: str) -> str:
    drift_note = ""
    if task_id == "task4_data_drift":
        drift_note = f"\nSTREAM ROW COUNT: {obs.get('row_count',{}).get('stream','?')}"
    return (
        f"Task: {obs['task_id']}\n"
        f"Step: {obs['step_count']}/{obs['max_steps']}\n"
        f"Score: {obs['partial_score']:.4f}\n"
        f"Last message: {obs['message']}\n"
        f"Schema errors: {obs['schema_errors'][:5]}\n"
        f"Column dtypes: {json.dumps(obs['column_dtypes'])}\n"
        f"Null counts: {json.dumps(obs['null_counts'])}\n"
        f"Duplicate counts: {obs['duplicate_count']}\n"
        f"Row counts: {obs['row_count']}\n"
        f"Available ops: {obs['available_operations']}"
        f"{drift_note}\n\nNext action JSON:"
    )


def run_episode(task_id: str, seed: int = 42) -> Tuple[str, float, float]:
    session_id = f"inference_{task_id}"
    client     = _make_client()
    t0         = time.time()
    max_steps  = TASK_MAX_STEPS[task_id]

    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "seed": seed, "session_id": session_id},
            timeout=30,
        )
        resp.raise_for_status()
        obs  = resp.json()
        done = obs.get("done", False)
    except Exception as e:
        print(f"  [{task_id}] Reset failed: {e}")
        return task_id, 0.0, 0.0

    print(f"  [{task_id}] started | score={obs['partial_score']:.3f}")

    for step_num in range(max_steps):
        if done:
            break

        try:
            prompt   = _build_prompt(obs, task_id)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw)
        except Exception as e:
            print(f"  [{task_id}] LLM error step {step_num+1}: {e} — submitting")
            action = {"operation": "submit"}

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step?session_id={session_id}",
                json=action,
                timeout=30,
            )
            step_resp.raise_for_status()
            data = step_resp.json()
            obs  = data["observation"]
            done = data["done"]
        except Exception as e:
            print(f"  [{task_id}] Step error: {e}")
            break

        print(f"  [{task_id}] step {step_num+1:2d} | op={action.get('operation'):20s} | "
              f"reward={data.get('reward', 0):+.4f} | score={obs['partial_score']:.4f}")
        time.sleep(0.3)

    elapsed = round(time.time() - t0, 2)
    score   = float(obs.get("partial_score", 0.0))
    print(f"  [{task_id}] FINAL score={score:.4f} | {elapsed}s")
    return task_id, score, elapsed


def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    try:
        h = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"Server health: {h.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {ENV_URL}: {e}")
        sys.exit(1)

    tasks = list(TASK_MAX_STEPS.keys())
    print(f"\n{'='*60}")
    print(f"  DataClean Inference | model={MODEL} | seed=42")
    print(f"  Running {len(tasks)} tasks SIMULTANEOUSLY")
    print(f"{'='*60}\n")

    t_total = time.time()
    scores:  Dict[str, float] = {}
    elapsed: Dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(run_episode, task_id, 42): task_id
            for task_id in tasks
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, score, secs = future.result()
                scores[tid]  = round(score, 4)
                elapsed[tid] = secs
            except Exception as exc:
                print(f"  [{task_id}] FAILED: {exc}")
                scores[task_id]  = 0.0
                elapsed[task_id] = -1.0

    wall_time = round(time.time() - t_total, 2)
    mean = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS (wall time: {wall_time}s)")
    print(f"{'='*60}")
    for k, v in scores.items():
        bar = "#" * int(v * 25)
        print(f"  {k:<25} {v:.4f}  {bar}")
    print(f"  {'mean':<25} {mean:.4f}")
    print(f"{'='*60}\n")
    print(json.dumps({**scores, "mean": mean, "wall_time_seconds": wall_time}, indent=2))


if __name__ == "__main__":
    main()