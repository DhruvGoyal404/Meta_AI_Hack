import os, json, time, sys
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

ENV_URL  = os.environ.get("ENV_URL", "http://localhost:7860")
API_KEY  = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def _make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

TASK_MAX_STEPS = {
    "task1": 10,
    "task2": 20,
    "task3": 30,
    "task4_data_drift": 40,
}

SYSTEM_PROMPT = """You are an expert data cleaning agent. Respond ONLY with valid JSON — no prose, no markdown.
... (unchanged prompt)
"""

def _build_prompt(obs: dict, task_id: str) -> str:
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
    session_id = f"baseline_{task_id}"
    client     = _make_client()
    t0         = time.time()
    max_steps  = TASK_MAX_STEPS[task_id]

    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed, "session_id": session_id},
        timeout=20,
    )
    resp.raise_for_status()
    obs  = resp.json()
    done = obs.get("done", False)

    # ✅ REQUIRED START BLOCK
    print(f"[START] task={task_id}", flush=True)

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
            print(f"  [{task_id}] LLM error step {step_num+1}: {e}", flush=True)
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

        # ✅ STEP BLOCK
        print(
            f"[STEP] task={task_id} step={step_num+1} "
            f"reward={data['reward']:.4f} score={obs['partial_score']:.4f}",
            flush=True
        )

        time.sleep(0.5)

    elapsed = round(time.time() - t0, 2)
    score   = float(obs.get("partial_score", 0.0))

    # ✅ END BLOCK
    print(
        f"[END] task={task_id} score={score:.4f} steps={step_num+1}",
        flush=True
    )

    return task_id, score, elapsed


def run_baseline_parallel(seed: int = 42) -> Dict[str, float]:
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    try:
        requests.get(f"{ENV_URL}/health", timeout=5)
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_URL}: {e}")
        sys.exit(1)

    tasks = list(TASK_MAX_STEPS.keys())

    # ✅ GLOBAL START
    print("[START] task=baseline", flush=True)

    t_total = time.time()
    scores: Dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(run_episode, task_id, seed): task_id
            for task_id in tasks
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, score, _ = future.result()
                scores[tid] = round(score, 4)
            except Exception as exc:
                print(f"[STEP] task={task_id} error={exc}", flush=True)
                scores[task_id] = 0.0

    wall_time = round(time.time() - t_total, 2)
    mean = round(sum(scores.values()) / len(scores), 4)

    # ✅ GLOBAL END
    print(
        f"[END] task=baseline score={mean:.4f} steps={len(scores)}",
        flush=True
    )

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
