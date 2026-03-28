"""
DataClean OpenEnv Client
Simple HTTP client — no openenv-core dependency needed.
"""
import json, requests
from typing import Any, Dict, Optional


class DataCleanEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860", session_id: str = "default"):
        self.base_url   = base_url.rstrip("/")
        self.session_id = session_id

    def reset(self, task_id: str = "task1", seed: int = 42) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/reset",
                          json={"task_id": task_id, "seed": seed, "session_id": self.session_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/step?session_id={self.session_id}", json=action)
        r.raise_for_status()
        return r.json()

    def state(self)    -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/state?session_id={self.session_id}").json()

    def tasks(self)    -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/tasks").json()

    def grader(self)   -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/grader?session_id={self.session_id}").json()

    def baseline(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/baseline", timeout=300).json()

    def health(self)   -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/health").json()


if __name__ == "__main__":
    env = DataCleanEnvClient()
    print("Health:", env.health())
    obs = env.reset("task1", seed=42)
    print("Tables:", list(obs["column_dtypes"].keys()))
    print("Nulls:", obs["null_counts"])
    print("Schema errors:", obs["schema_errors"])