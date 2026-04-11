"""
Parallel Baseline Script
========================
Runs all 4 tasks SIMULTANEOUSLY using concurrent.futures.ThreadPoolExecutor.
Each task gets its own thread + its own session_id — fully isolated.

Wall-clock time = time of slowest task (not sum of all tasks).
On a 4-task run: ~3x faster than sequential.

Usage:
    # Terminal 1: uvicorn server.app:app --host 0.0.0.0 --port 7860
    # Terminal 2:
    $env:OPENAI_API_KEY = "gsk_..."
    $env:OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
    $env:BASELINE_MODEL = "llama-3.3-70b-versatile"
    python baseline.py
"""
import os, json, time, sys
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

ENV_URL  = os.environ.get("ENV_URL", "http://localhost:7860")
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")

# Each task runs in its own thread with its own OpenAI client (thread-safe)
def _make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

TASK_MAX_STEPS = {
    "task1": 10,
    "task2": 20,
    "task3": 30,
    "task4_data_drift": 40,
}

SYSTEM_PROMPT = """You are an expert data cleaning agent. Respond ONLY with valid JSON — no prose, no markdown.

Operations:
  fill_nulls:         {"operation":"fill_nulls","column":"<col>","strategy":"mean|median|mode|constant|forward_fill|backward_fill","table_name":"<tbl>"}
  cast_column:        {"operation":"cast_column","column":"<col>","dtype":"int|float|str|datetime","table_name":"<tbl>"}
  remove_duplicates:  {"operation":"remove_duplicates","table_name":"<tbl>"}
  normalize_values:   {"operation":"normalize_values","column":"<col>","method":"upper|lower|regex","table_name":"<tbl>"}
  filter_outliers:    {"operation":"filter_outliers","column":"<col>","method":"iqr|zscore","threshold":1.5,"table_name":"<tbl>"}
  merge_tables:       {"operation":"merge_tables","left_table":"orders","right_table":"customers","on":"customer_id","output_table":"merged"}
  add_derived_column: {"operation":"add_derived_column","column_name":"order_year","source_column":"order_date","transform":"year_from_date","table_name":"merged"}
  submit:             {"operation":"submit"}

Task strategies:
  task1: fill_nulls(age,median,main)→cast_column(age,int,main)→fill_nulls(salary,mean,main)→submit
  task2: remove_duplicates(main)→normalize_values(country,upper,main)→cast_column(order_date,datetime,main)→fill_nulls(amount,mean,main)→submit
  task3: merge_tables(orders,customers,customer_id)→fill_nulls(age,median,merged)→cast_column(age,int,merged)→filter_outliers(amount,iqr,1.5,merged)→add_derived_column(order_year,order_date,year_from_date,merged)→submit
  task4_data_drift: filter_outliers(amount,iqr,1.5,stream)→fill_nulls(amount,mean,stream)→cast_column(amount,float,stream)→fill_nulls(category,mode,stream)→fill_nulls(region,mode,stream)→cast_column(event_ts,datetime,stream)→[repeat after each drift injection]→submit

IMPORTANT for task4: New dirty rows are injected every 5 steps automatically.
After each drift injection (shown in message), re-run your cleaning ops on 'stream'.
"""


def _build_prompt(obs: dict, task_id: str) -> str:
    # For task4 show drift-specific info
    drift_note = ""
    if task_id == "task4_data_drift":
        drift_note = f"\nDRIFT TABLE ROW COUNT: {obs.get('row_count',{}).get('stream','?')}"
        drift_note += f"\n[Watch message for drift injections — re-clean after each one]"

    return (
        f"Task: {obs['task_id']}\n"
        f"Step: {obs['step_count']}/{obs['max_steps']}\n"
        f"Score: {obs['partial_score']:.4f}\n"
        f"Last message: {obs['message']}\n"
        f"Schema errors: {obs['schema_errors'][:6]}\n"
        f"Column dtypes: {json.dumps(obs['column_dtypes'])}\n"
        f"Null counts: {json.dumps(obs['null_counts'])}\n"
        f"Duplicate counts: {obs['duplicate_count']}\n"
        f"Row counts: {obs['row_count']}\n"
        f"Available ops: {obs['available_operations']}"
        f"{drift_note}\n\nNext action JSON:"
    )


def run_episode(task_id: str, seed: int = 42) -> Tuple[str, float, float]:
    """
    Run one full episode for task_id.
    Returns (task_id, final_score, elapsed_seconds).
    Each call is self-contained — uses its own session_id and OpenAI client.
    """
    session_id = f"baseline_{task_id}"
    client     = _make_client()
    t0         = time.time()
    max_steps  = TASK_MAX_STEPS[task_id]

    # Reset
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed, "session_id": session_id},
        timeout=20,
    )
    resp.raise_for_status()
    obs  = resp.json()
    done = obs.get("done", False)

    print(f"  [{task_id}] started | score={obs['partial_score']:.3f} | "
          f"tables={list(obs['column_dtypes'].keys())}")

    for step_num in range(max_steps):
        if done:
            break

        prompt = _build_prompt(obs, task_id)
        try:
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
                if raw.startswith("json"): raw = raw[4:]
            action = json.loads(raw)
        except Exception as e:
            print(f"  [{task_id}] LLM error step {step_num+1}: {e} — submitting")
            action = {"operation": "submit"}

        step_resp = requests.post(
            f"{ENV_URL}/step?session_id={session_id}",
            json=action,
            timeout=20,
        )
        step_resp.raise_for_status()
        data = step_resp.json()
        obs  = data["observation"]
        done = data["done"]

        print(f"  [{task_id}] step {step_num+1:2d} | op={action.get('operation'):22s} | "
              f"reward={data['reward']:+.4f} | score={obs['partial_score']:.4f}")
        time.sleep(0.5)  # light rate-limit buffer

    elapsed = round(time.time() - t0, 2)
    score   = float(obs.get("partial_score", 0.0))
    print(f"  [{task_id}] DONE | score={score:.4f} | {elapsed}s")
    return task_id, score, elapsed


def run_baseline_parallel(seed: int = 42) -> Dict[str, float]:
    """
    Run all tasks IN PARALLEL using ThreadPoolExecutor.
    Wall-clock time = slowest task, not sum of all tasks.
    """
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    try:
        h = requests.get(f"{ENV_URL}/health", timeout=5)
        print(f"Server: {h.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_URL}: {e}"); sys.exit(1)

    tasks = list(TASK_MAX_STEPS.keys())

    print(f"\n{'='*60}")
    print(f"  DataClean Parallel Baseline")
    print(f"  model={MODEL} | seed={seed} | tasks={tasks}")
    print(f"  Running {len(tasks)} tasks SIMULTANEOUSLY (ThreadPoolExecutor)")
    print(f"{'='*60}\n")

    t_total = time.time()
    scores: Dict[str, float] = {}
    elapsed: Dict[str, float] = {}

    # All tasks run in parallel — each thread is fully independent
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(run_episode, task_id, seed): task_id
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
                scores[task_id]  = 0.001
                elapsed[task_id] = -1.0

    wall_time = round(time.time() - t_total, 2)
    mean = round(sum(scores.values()) / len(scores), 4)

    print(f"\n{'='*60}")
    print(f"  RESULTS  (wall time: {wall_time}s)")
    print(f"{'='*60}")
    for k, v in scores.items():
        bar  = "#" * int(v * 25)
        diff = elapsed.get(k, 0)
        print(f"  {k:<25} {v:.4f}  {bar:<25} ({diff}s)")
    print(f"  {'mean':<25} {mean:.4f}")
    print(f"{'='*60}\n")
    print(json.dumps({**scores, "mean": mean, "wall_time_seconds": wall_time}, indent=2))
    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--url",  default="http://localhost:7860")
    args = parser.parse_args()
    ENV_URL = args.url
    run_baseline_parallel(seed=args.seed)
