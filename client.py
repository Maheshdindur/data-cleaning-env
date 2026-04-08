from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, Optional

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DataCleaningEnv:
    """
    Sync client for the Data Cleaning OpenEnv environment.

    Usage:
        env = DataCleaningEnv(base_url="http://localhost:8000")
        obs = env.reset(task_id="easy")
        result = env.step({"operation": "fill_nulls", "column": "age",
                           "params": {"strategy": "median"}, "reasoning": "age has nulls"})
        print(result["observation"]["data_quality_score"])
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        if not HAS_REQUESTS:
            raise ImportError("pip install requests")

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # Convenience: run a full episode with a simple policy function
    def run_episode(self, policy_fn, task_id: str = "easy") -> Dict[str, Any]:
        """
        policy_fn(observation: dict) -> action: dict
        Returns episode summary dict.
        """
        result = self.reset(task_id=task_id)
        obs    = result["observation"]
        total_reward = 0.0
        steps = 0

        while not obs["done"]:
            action = policy_fn(obs)
            result = self.step(action)
            obs    = result["observation"]
            if obs["reward"] is not None:
                total_reward += obs["reward"]
            steps += 1

        return {
            "task_id":       task_id,
            "steps":         steps,
            "total_reward":  round(total_reward, 4),
            "final_quality": obs["data_quality_score"],
            "done":          obs["done"],
        }


class AsyncDataCleaningEnv:
    """Async WebSocket client — preferred for high-throughput training."""

    def __init__(self, base_url: str = "ws://localhost:8000"):
        if not HAS_WS:
            raise ImportError("pip install websockets")
        self.ws_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws = None

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url)
        return self

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        await self._ws.send(json.dumps({"method": "reset", "payload": {"task_id": task_id}}))
        return json.loads(await self._ws.recv())

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"method": "step", "payload": action}))
        return json.loads(await self._ws.recv())

    async def state(self) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"method": "state", "payload": {}}))
        return json.loads(await self._ws.recv())

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *args):
        await self.close()
