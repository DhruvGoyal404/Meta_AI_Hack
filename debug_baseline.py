import os
import json
import requests
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
)

ENV_URL = "http://localhost:7860"

def debug_episode(task_id: str, seed: int = 42):
    # Reset
    reset_data = {"task_id": task_id, "seed": seed}
    print(f"Resetting {task_id}...")
    resp = requests.post(f"{ENV_URL}/reset", json=reset_data)
    obs = resp.json()
    print(f"Initial partial_score: {obs['partial_score']}")
    
    # Make one LLM call
    prompt = f"""You are a data cleaning assistant. Current state:
Task: {obs['task_description']}
Step: {obs['step_count']}/{obs['max_steps']}
Current score: {obs['partial_score']:.3f}

Data summary (first few rows of main table):
{obs['tables'].get('main', '')[:500]}

Respond with a JSON action, e.g., {{"operation": "submit"}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    action_text = response.choices[0].message.content
    print("LLM response:", action_text)
    try:
        action = json.loads(action_text)
        print("Parsed action:", action)
    except Exception as e:
        print("Failed to parse JSON:", e)

if __name__ == "__main__":
    for task in ["task1", "task2", "task3"]:
        print(f"\n=== Debugging {task} ===")
        debug_episode(task, seed=42)