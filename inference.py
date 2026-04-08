import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ENV CONFIG
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 8
MAX_TOKENS = 150
TEMPERATURE = 0.7

SUCCESS_SCORE_THRESHOLD = 0.1

# reward normalization
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = """You are interacting with an environment. Respond with a useful message each step."""

# ---------------- LOGGING (STRICT FORMAT) ---------------- #

def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# ---------------- MODEL ---------------- #

def get_model_message(client, step, history):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}. Continue interaction."},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content.strip() or "hello"
    except:
        return "hello"

# ---------------- MAIN ---------------- #

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards = []
    history = []

    steps_taken = 0
    success = False
    score = 0.0

    log_start()

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_message(client, step, history)

            result = await env.step(MyEnvV4Action(message=action))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, None)

            history.append(action)

            if done:
                break

        # score calculation
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
