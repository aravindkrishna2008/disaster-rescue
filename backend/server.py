"""
FastAPI backend for the Battle Angel console.

POST /command  body: {"text": "<operator order>"}
returns:       {target_id, confidence, reason, reached, steps, total_reward, gif_url}

Pipeline:
  1. Gemini picks which survivor ("child" | "adult") to rescue from the order
  2. Build a scene with that survivor's coords and run one PPO episode
  3. Return aggregate stats; the rendered GIF is exposed at /episode.gif

Episode rollouts are CPU-bound (PPO inference + MuJoCo), so we run them on a
thread via asyncio.to_thread to keep the event loop responsive.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from export_run import list_runs, load_run_manifest, manifest_to_frontend
from disaster_env import DEFAULT_SCENE
from gemini_client import SURVIVORS, get_gemini_target
from rescue_runner import run_episode
from scenes import GENERATED_SCENES, get_scene

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_OUTPUT = _HERE / "output"
_RUNS = _HERE / "runs"
_OUTPUT.mkdir(exist_ok=True)
_RUNS.mkdir(exist_ok=True)

MODEL_PATH = os.environ.get(
    "BATTLE_ANGEL_MODEL", str(_HERE / "models" / "ppo_buried_detection_final")
)
MAX_STEPS = int(os.environ.get("BATTLE_ANGEL_MAX_STEPS", "300"))

# Step presets surfaced to the frontend dropdown — these are PPO training
# timesteps. Selecting one and pressing Train kicks off a fresh training run.
STEP_PRESETS = [
    {"label": "50k", "value": 50_000},
    {"label": "100k", "value": 100_000},
    {"label": "200k", "value": 200_000},
    {"label": "500k", "value": 500_000},
    {"label": "1M", "value": 1_000_000},
]
DEFAULT_TRAIN_STEPS = 200_000
FIXED_NUM_ENVS = 8

# Track cancel events for active episode runs so the frontend can kill them.
_ACTIVE_CANCELS: dict[str, threading.Event] = {}
_ACTIVE_LOCK = threading.Lock()


def _register_cancel(key: str) -> threading.Event:
    ev = threading.Event()
    with _ACTIVE_LOCK:
        existing = _ACTIVE_CANCELS.get(key)
        if existing is not None:
            existing.set()
        _ACTIVE_CANCELS[key] = ev
    return ev


def _release_cancel(key: str, ev: threading.Event) -> None:
    with _ACTIVE_LOCK:
        if _ACTIVE_CANCELS.get(key) is ev:
            _ACTIVE_CANCELS.pop(key, None)


def _cancel_all() -> int:
    with _ACTIVE_LOCK:
        events = list(_ACTIVE_CANCELS.values())
        _ACTIVE_CANCELS.clear()
    for ev in events:
        ev.set()
    return len(events)


# ----- Training subprocess management ---------------------------------------
_TRAIN_LOCK = threading.Lock()
_TRAIN_STATE: dict = {
    "status": "idle",      # idle | running | done | error | cancelled
    "total_steps": 0,
    "started_at": None,
    "finished_at": None,
    "pid": None,
    "exit_code": None,
    "error": None,
}
_TRAIN_PROC: subprocess.Popen | None = None


def _train_worker(total_steps: int) -> None:
    global _TRAIN_PROC
    env = {**os.environ, "BATTLE_ANGEL_TRAIN_STEPS": str(total_steps)}
    try:
        proc = subprocess.Popen(
            [sys.executable, str(_HERE / "train.py")],
            cwd=str(_HERE),
            env=env,
            # Own process group so we can SIGTERM the trainer AND its
            # SubprocVecEnv forkserver children together — otherwise the
            # orphaned forkserver crashes with BrokenPipeError on shutdown.
            start_new_session=True,
        )
    except Exception as e:
        with _TRAIN_LOCK:
            _TRAIN_STATE["status"] = "error"
            _TRAIN_STATE["error"] = str(e)
            _TRAIN_STATE["finished_at"] = time.time()
        return

    with _TRAIN_LOCK:
        _TRAIN_PROC = proc
        _TRAIN_STATE["pid"] = proc.pid

    exit_code = proc.wait()

    with _TRAIN_LOCK:
        _TRAIN_PROC = None
        _TRAIN_STATE["exit_code"] = exit_code
        _TRAIN_STATE["finished_at"] = time.time()
        if _TRAIN_STATE["status"] == "cancelled":
            pass
        elif exit_code == 0:
            _TRAIN_STATE["status"] = "done"
        else:
            _TRAIN_STATE["status"] = "error"
            _TRAIN_STATE["error"] = f"train.py exited with code {exit_code}"


def _start_training(total_steps: int) -> dict:
    with _TRAIN_LOCK:
        if _TRAIN_STATE["status"] == "running":
            raise RuntimeError("training already in progress")
        _TRAIN_STATE.update(
            status="running",
            total_steps=total_steps,
            started_at=time.time(),
            finished_at=None,
            pid=None,
            exit_code=None,
            error=None,
        )
    threading.Thread(target=_train_worker, args=(total_steps,), daemon=True).start()
    return dict(_TRAIN_STATE)


def _stop_training() -> bool:
    global _TRAIN_PROC
    with _TRAIN_LOCK:
        proc = _TRAIN_PROC
        if proc is None or _TRAIN_STATE["status"] != "running":
            return False
        _TRAIN_STATE["status"] = "cancelled"

    # Kill the entire process group so SubprocVecEnv's forkserver and worker
    # processes die alongside the trainer (avoids orphaned-forkserver
    # BrokenPipeError tracebacks).
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _escalate() -> None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    threading.Thread(target=_escalate, daemon=True).start()
    return True


class CommandRequest(BaseModel):
    text: str


class SceneRunRequest(BaseModel):
    max_steps: int | None = None


class TrainRequest(BaseModel):
    total_steps: int | None = None


def _scene_summary(idx: int, scene: dict) -> dict:
    return {
        "index": idx,
        "name": scene["name"],
        "difficulty": scene.get("difficulty", "medium"),
        "robot_start": scene.get("robot_start"),
        "survivor_pos": scene.get("survivor_pos"),
        "survivor": scene.get("survivor", {}),
        "obstacle_count": len(scene.get("obstacles", [])),
        "hazard_count": len(scene.get("hazards", [])),
        "obstacles": scene.get("obstacles", []),
        "hazards": scene.get("hazards", []),
    }


def _gif_filename_for(idx: int, name: str) -> str:
    return f"{idx + 1:02d}_{name}.gif"


def _run_scene(idx: int, max_steps: int, cancel_event: threading.Event) -> dict:
    scene = get_scene(idx)
    name = scene["name"]
    gif_name = _gif_filename_for(idx, name)
    gif_path = _OUTPUT / gif_name
    result = run_episode(
        scene=scene,
        model_path=MODEL_PATH,
        gif_path=str(gif_path),
        max_steps=max_steps,
        cancel_event=cancel_event,
    )
    return {
        "scene_index": idx,
        "scene_name": name,
        "difficulty": scene.get("difficulty", "medium"),
        "reached": bool(result.get("reached", False)),
        "steps": int(result.get("steps", 0)),
        "total_reward": float(result.get("total_reward", 0.0)),
        "trajectory": result.get("trajectory", []),
        "detection_event": result.get("detection_event"),
        "cancelled": bool(result.get("cancelled", False)),
        "survivor": scene.get("survivor", {}),
        "gif_url": f"/gifs/{gif_name}",
        "max_steps": max_steps,
    }


def _scene_for_target(target_id: str) -> dict:
    """Clone DEFAULT_SCENE but plant the survivor at the chosen target coords."""
    if target_id not in SURVIVORS:
        raise ValueError(f"unknown target_id: {target_id!r}")
    x, y = SURVIVORS[target_id]
    return {
        **DEFAULT_SCENE,
        "survivor_pos": [float(x), float(y), 0.0],
    }


def _run_episode_for(target_id: str, cancel_event: threading.Event) -> dict:
    scene = _scene_for_target(target_id)
    gif_path = _OUTPUT / f"episode_{target_id}.gif"
    return run_episode(
        scene=scene,
        model_path=MODEL_PATH,
        gif_path=str(gif_path),
        max_steps=MAX_STEPS,
        cancel_event=cancel_event,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Battle Angel · Ground Rescue")

    # Allow direct calls from the Next.js dev server (the rewrite in
    # next.config.mjs hides this in prod, but CORS makes ad-hoc testing easy).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return (_STATIC / "index.html").read_text()

    app.mount("/gifs", StaticFiles(directory=_OUTPUT), name="gifs")
    app.mount("/run-gifs", StaticFiles(directory=_RUNS), name="run-gifs")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": MODEL_PATH, "max_steps": MAX_STEPS}

    @app.get("/episode.gif")
    def latest_gif():
        gifs = sorted(
            _OUTPUT.glob("episode_*.gif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not gifs:
            raise HTTPException(status_code=404, detail="no episode gif yet")
        return FileResponse(gifs[0], media_type="image/gif")

    @app.get("/runs")
    def get_runs() -> dict:
        return {"runs": list_runs()}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        manifest = load_run_manifest(run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return manifest_to_frontend(manifest)

    @app.get("/scenes")
    def list_scenes() -> dict:
        scenes = [_scene_summary(i, scene) for i, scene in enumerate(GENERATED_SCENES)]
        return {
            "scenes": scenes,
            "default_max_steps": MAX_STEPS,
            "step_presets": STEP_PRESETS,
            "default_train_steps": DEFAULT_TRAIN_STEPS,
            "num_envs": FIXED_NUM_ENVS,
        }

    @app.post("/runs/cancel")
    def cancel_runs() -> dict:
        cancelled = _cancel_all()
        train_stopped = _stop_training()
        return {"cancelled": cancelled, "training_stopped": train_stopped}

    @app.get("/train/status")
    def train_status() -> dict:
        with _TRAIN_LOCK:
            return dict(_TRAIN_STATE)

    @app.post("/train")
    def start_train(req: TrainRequest) -> dict:
        steps = req.total_steps or DEFAULT_TRAIN_STEPS
        if steps <= 0:
            raise HTTPException(status_code=400, detail="total_steps must be positive")
        try:
            return _start_training(steps)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.post("/train/cancel")
    def cancel_train() -> dict:
        return {"stopped": _stop_training()}

    @app.post("/scene/{idx}/run")
    async def run_scene(idx: int, req: SceneRunRequest) -> dict:
        if idx < 0 or idx >= len(GENERATED_SCENES):
            raise HTTPException(status_code=404, detail=f"scene {idx} not found")

        max_steps = req.max_steps or MAX_STEPS
        key = f"scene:{idx}"
        cancel_event = _register_cancel(key)
        try:
            result = await asyncio.to_thread(_run_scene, idx, max_steps, cancel_event)
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"model not found: {e}") from e
        finally:
            _release_cancel(key, cancel_event)

    @app.post("/command")
    async def command(req: CommandRequest) -> dict:
        decision = get_gemini_target(req.text, SURVIVORS)
        target_id = decision["target_id"]

        key = f"command:{target_id}"
        cancel_event = _register_cancel(key)
        try:
            result = await asyncio.to_thread(_run_episode_for, target_id, cancel_event)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"model not found: {e}") from e
        finally:
            _release_cancel(key, cancel_event)

        return {
            "target_id": target_id,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason": decision.get("reason", ""),
            "reached": bool(result.get("reached", False)),
            "steps": int(result.get("steps", 0)),
            "total_reward": float(result.get("total_reward", 0.0)),
            "gif_url": "/episode.gif",
            "trajectory": result.get("trajectory", []),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
