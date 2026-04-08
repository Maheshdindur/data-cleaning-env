"""
Baseline inference script for the Data Cleaning Environment.

Reads OPENAI_API_KEY (or GROQ_API_KEY with OPENAI_BASE_URL) from environment.
Runs the base model against all 3 tasks and prints reproducible scores.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

# 1. FORCE load_dotenv and fail loudly if missing
try:
    from dotenv import load_dotenv
    # override=True ensures the .env file overrides any existing system variables
    load_dotenv(override=True)
except ImportError:
    print("ERROR: The 'python-dotenv' package is not installed.")
    print("Please run this command first: pip install python-dotenv")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: The 'openai' package is not installed.")
    print("Please run: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: The 'requests' package is not installed.")
    print("Please run: pip install requests")
    sys.exit(1)

# 2. Verify the API key was actually loaded
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    print("ERROR: OPENAI_API_KEY is missing!")
    print("Please make sure you have a .env file in this folder with OPENAI_API_KEY=your_key")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL        = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
EPISODES     = int(os.environ.get("BASELINE_EPISODES", "5"))   # per task
MAX_STEPS    = 20

client = OpenAI(
    api_key  = api_key,
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert data engineer. You are given a dirty dataset and must clean it
step by step using the available operations.

Available operations:
- fill_nulls(column, params={strategy: mean|median|mode|value})
- drop_duplicates()
- fix_type(column, params={dtype: int|float|str})
- normalise_column(column, params={mapping: {old: new, ...}})
- drop_column(column)   -- only for truly irrelevant columns
- fix_date_format(column)
- clip_outliers(column, params={lower_pct: 5, upper_pct: 95})
- rename_column(column, params={new_name: "..."})
- submit()              -- call when you believe the dataset is clean

Respond ONLY with valid JSON in this exact format:
{
  "operation": "<operation_name>",
  "column": "<column_name_or_null>",
  "params": {},
  "reasoning": "<one sentence explaining why>"
}
""".strip()


def format_observation(obs: Dict[str, Any]) -> str:
    lines = [
        f"Task: {obs['task_id']} | Step: {obs['step']} | Quality: {obs['data_quality_score']:.3f}",
        f"Shape: {obs['shape'][0]} rows × {obs['shape'][1]} cols",
        "",
        "Columns:",
    ]
    for col in obs["columns"]:
        lines.append(
            f"  {col['name']} ({col['dtype']}) — "
            f"{col['null_count']} nulls, {col['unique_count']} unique — "
            f"samples: {col['sample_values'][:3]}"
        )
    lines.append("")
    lines.append("Issues detected:")
    for issue in obs["issues_detected"]:
        lines.append(f"  • {issue}")
    lines.append(f"\nLast message: {obs['message']}")
    return "\n".join(lines)


def get_action(obs: Dict[str, Any], history: list) -> Dict[str, Any]:
    """Ask the LLM for the next cleaning action — retries on rate limit."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": "user",      "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({"role": "user", "content": format_observation(obs)})

    for attempt in range(3):   # retry up to 3 times
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "429" in err:
                wait = (attempt + 1) * 10
                print(f"\n  [rate limit] waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  [LLM error] {exc} — defaulting to submit")
                return {"operation": "submit", "column": None, "params": {}, "reasoning": "fallback"}

    print("  [max retries] defaulting to submit")
    return {"operation": "submit", "column": None, "params": {}, "reasoning": "fallback"}


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run one full episode against the live environment server."""
    # Reset
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs  = resp.json()["observation"]

    history      = []
    total_reward = 0.0
    steps        = 0

    while not obs["done"] and steps < MAX_STEPS:
        action      = get_action(obs, history)
        action_str  = json.dumps(action)

        step_resp   = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result      = step_resp.json()
        new_obs     = result["observation"]

        history.append({"obs": format_observation(obs), "action": action_str})
        reward = new_obs.get("reward") or 0.0
        total_reward += reward
        steps += 1
        obs = new_obs

        time.sleep(1.0)   # Groq free tier: ~30 req/min, 1s gap is safe

    return {
        "task_id":       task_id,
        "steps":         steps,
        "total_reward":  round(total_reward, 4),
        "final_quality": obs["data_quality_score"],
    }


def main():
    print(f"{'='*60}")
    print(f"Data Cleaning Environment — Baseline Inference")
    print(f"Model: {MODEL} | Episodes per task: {EPISODES}")
    print(f"Environment: {BASE_URL}")
    print(f"{'='*60}\n")

    # Verify environment is running
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=5)
        assert h.status_code == 200
        print("✓ Environment healthy\n")
    except Exception:
        print(f"✗ Cannot reach environment at {BASE_URL}")
        print("  Start it with: uvicorn server.app:app --port 8000")
        sys.exit(1)

    all_results = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"── Task: {task_id.upper()} {'─'*40}")
        episodes = []
        for ep in range(1, EPISODES + 1):
            print(f"  Episode {ep}/{EPISODES} ... ", end="", flush=True)
            result = run_episode(task_id)
            episodes.append(result)
            print(f"quality={result['final_quality']:.3f}  steps={result['steps']}")

        avg_quality = sum(e["final_quality"] for e in episodes) / len(episodes)
        avg_reward  = sum(e["total_reward"]  for e in episodes) / len(episodes)
        all_results[task_id] = {
            "avg_quality": round(avg_quality, 4),
            "avg_reward":  round(avg_reward,  4),
            "episodes":    episodes,
        }
        print(f"  → Average quality: {avg_quality:.4f}  |  Avg reward: {avg_reward:.4f}\n")

    # Summary
    print(f"{'='*60}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    for task_id, res in all_results.items():
        print(f"  {task_id.upper():<8}  quality={res['avg_quality']:.4f}  reward={res['avg_reward']:.4f}")

    overall = sum(r["avg_quality"] for r in all_results.values()) / 3
    print(f"\n  OVERALL   {overall:.4f}")
    print(f"{'='*60}")

    # Save results for README
    with open("baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDetailed results saved to baseline_results.json")


if __name__ == "__main__":
    main()