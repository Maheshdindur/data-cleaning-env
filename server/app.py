from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from server.environment import DataCleaningEnvironment

app = FastAPI(
    title="Data Cleaning Environment",
    description="OpenEnv-compliant environment for training AI agents on real-world data cleaning tasks.",
    version="1.0.0",
)

# One environment instance per WebSocket session
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/info")
async def info():
    return {
        "name": "data-cleaning-env",
        "tasks": ["easy", "medium", "hard"],
        "actions": [
            "fill_nulls", "drop_duplicates", "fix_type",
            "normalise_column", "drop_column", "fix_date_format",
            "clip_outliers", "rename_column", "submit",
        ],
        "spec_version": "1.0",
    }


@app.post("/reset")
async def reset(body: Dict[str, Any] = {}):
    env = DataCleaningEnvironment()
    task_id = body.get("task_id", "easy")
    obs = env.reset(task_id=task_id)
    # Store env in app state keyed by a simple counter (demo — use session mgmt for prod)
    app.state.env = env
    return JSONResponse(content={"observation": obs, "state": env.state})


@app.post("/step")
async def step(action: Dict[str, Any]):
    env = getattr(app.state, "env", None)
    if env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first"})
    result = env.step(action)
    return JSONResponse(content={"observation": result, "state": env.state})


@app.get("/state")
async def state():
    env = getattr(app.state, "env", None)
    if env is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first"})
    return JSONResponse(content=env.state)


# WebSocket endpoint — primary interface (low latency, persistent session)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = DataCleaningEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            method  = msg.get("method")
            payload = msg.get("payload", {})

            if method == "reset":
                task_id = payload.get("task_id", "easy")
                obs = env.reset(task_id=task_id)
                await websocket.send_text(json.dumps({"observation": obs, "state": env.state}))

            elif method == "step":
                result = env.step(payload)
                await websocket.send_text(json.dumps({"observation": result, "state": env.state}))

            elif method == "state":
                await websocket.send_text(json.dumps({"state": env.state}))

            else:
                await websocket.send_text(json.dumps({"error": f"Unknown method '{method}'"}))

    except WebSocketDisconnect:
        pass
