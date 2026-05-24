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
from copy import deepcopy
import os
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from disaster_env import DEFAULT_BALANCE_ASSIST_SCALE, DEFAULT_SCENE, generate_random_terrain
from export_run import list_runs, load_run_manifest, manifest_to_frontend
from run_store import current_run_id, append_live_eval, run_dir, read_json, load_episode
from episode_loader import replay_episode_actions
from gemini_client import SURVIVORS, generate_scene_prompt, get_gemini_target
from rescue_runner import run_episode
from scenes import GENERATED_SCENES, get_scene
from scenario_agent import (
    Difficulty,
    ScenarioAgent,
    SceneSpec,
    build_generation_prompt,
    normalize_survivor_profiles,
)
from scene_adapter import scene_spec_to_env_scene

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_OUTPUT = _HERE / "output"
_RUNS = _HERE / "runs"
_OUTPUT.mkdir(exist_ok=True)
_RUNS.mkdir(exist_ok=True)

MODEL_PATH = os.environ.get(
    "BATTLE_ANGEL_MODEL", str(_HERE / "models" / "g1_locomotion_natural_final")
)
MAX_STEPS = int(os.environ.get("BATTLE_ANGEL_MAX_STEPS", "900"))
GENERATED_MAX_STEPS = int(os.environ.get("BATTLE_ANGEL_GENERATED_MAX_STEPS", "1500"))
MAX_REQUESTED_EPISODE_STEPS = int(os.environ.get("BATTLE_ANGEL_MAX_REQUESTED_STEPS", "3000"))
ASSIST_SCALE = float(os.environ.get("BATTLE_ANGEL_ASSIST_SCALE", "0.95"))
BALANCE_ASSIST_SCALE = float(
    os.environ.get("BATTLE_ANGEL_BALANCE_ASSIST_SCALE", str(DEFAULT_BALANCE_ASSIST_SCALE))
)

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
_EPISODE_CONCURRENCY = max(1, int(os.environ.get("BATTLE_ANGEL_EPISODE_CONCURRENCY", "2")))
_EPISODE_SEMAPHORE = threading.Semaphore(_EPISODE_CONCURRENCY)
_GENERATED_SESSIONS: dict[str, dict] = {}
_GENERATED_LOCK = threading.Lock()


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


def _episode_steps(requested: int | None, default: int) -> int:
    steps = default if requested is None else int(requested)
    if steps <= 0 or steps > MAX_REQUESTED_EPISODE_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"max_steps must be between 1 and {MAX_REQUESTED_EPISODE_STEPS}",
        )
    return steps


def _save_generated_session(payload: dict) -> dict:
    scene_id = payload["scene_id"]
    with _GENERATED_LOCK:
        _GENERATED_SESSIONS[scene_id] = deepcopy(payload)
        return deepcopy(_GENERATED_SESSIONS[scene_id])


def _load_generated_session(scene_id: str) -> dict:
    with _GENERATED_LOCK:
        payload = _GENERATED_SESSIONS.get(scene_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"generated scene {scene_id!r} not found")
        return deepcopy(payload)


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
            [sys.executable, str(_HERE / "train.py"), "--timesteps", str(total_steps)],
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


class GeneratePromptRequest(BaseModel):
    difficulty: Difficulty = "medium"
    survivor_count: int = 1
    theme: str | None = None


class GenerateSceneRequest(BaseModel):
    description: str
    difficulty: Difficulty = "medium"
    survivor_count: int = 1
    theme: str | None = None
    skip_episode: bool = False
    max_steps: int | None = None


def _scene_summary(idx: int, scene: dict) -> dict:
    difficulty = scene.get("difficulty", "medium")
    return {
        "index": idx,
        "name": scene["name"],
        "difficulty": difficulty,
        "robot_start": scene.get("robot_start"),
        "survivor_pos": scene.get("survivor_pos"),
        "survivor": scene.get("survivor", {}),
        "obstacle_count": len(scene.get("obstacles", [])),
        "hazard_count": len(scene.get("hazards", [])),
        "obstacles": [_decorate_obstacle(obstacle, i) for i, obstacle in enumerate(scene.get("obstacles", []))],
        "hazards": [_decorate_hazard(hazard, i) for i, hazard in enumerate(scene.get("hazards", []))],
        "terrain": scene.get("terrain") or generate_random_terrain(
            seed=idx + 1000,
            difficulty=difficulty,
            grid_size=10,
        ),
    }


def _decorate_obstacle(obstacle: dict, index: int) -> dict:
    colors = ["#8e9294", "#5f6870", "#7b6f61", "#a89f93", "#6d6258"]
    return {**obstacle, "color": obstacle.get("color", colors[index % len(colors)])}


def _decorate_hazard(hazard: dict, index: int) -> dict:
    hazard_type = hazard.get("type")
    colors = {
        "fire": "#ef3b2d",
        "gas": "#2fbf71",
        "flood": "#2f80ed",
        "unstable_floor": "#f2b705",
    }
    fallback = ["#ef3b2d", "#2fbf71", "#f2b705"]
    return {
        **hazard,
        "type": hazard_type or ("fire" if index % 2 == 0 else "gas"),
        "color": hazard.get("color", colors.get(hazard_type, fallback[index % len(fallback)])),
    }


def _gif_filename_for(idx: int, name: str) -> str:
    return f"{idx + 1:02d}_{name}.gif"


def _run_scene(idx: int, max_steps: int, cancel_event: threading.Event) -> dict:
    with _EPISODE_SEMAPHORE:
        scene = get_scene(idx)
        name = scene["name"]
        gif_name = _gif_filename_for(idx, name)
        gif_path = _OUTPUT / gif_name
        result = run_episode(
            scene=scene,
            model_path=MODEL_PATH,
            gif_path=str(gif_path),
            max_steps=max_steps,
            assist_scale=ASSIST_SCALE,
            balance_assist_scale=BALANCE_ASSIST_SCALE,
            cancel_event=cancel_event,
        )
        return {
            "scene_index": idx,
            "scene_name": name,
            "difficulty": scene.get("difficulty", "medium"),
            "reached": bool(result.get("reached", False)),
            "fallen": bool(result.get("fallen", False)),
            "steps": int(result.get("steps", 0)),
            "total_reward": float(result.get("total_reward", 0.0)),
            "final_dist": result.get("final_dist"),
            "min_dist": result.get("min_dist"),
            "obstacle_contacts": result.get("obstacle_contacts", 0),
            "hazard_steps": result.get("hazard_steps", 0),
            "mean_stance_slip": result.get("mean_stance_slip", 0.0),
            "mean_assist_force": result.get("mean_assist_force", 0.0),
            "gait_score": result.get("gait_score", 0.0),
            "assist_scale": result.get("assist_scale", ASSIST_SCALE),
            "balance_assist_scale": result.get("balance_assist_scale", BALANCE_ASSIST_SCALE),
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
        assist_scale=ASSIST_SCALE,
        balance_assist_scale=BALANCE_ASSIST_SCALE,
        cancel_event=cancel_event,
    )


def _score_generated_scene(scene: SceneSpec, env_scene: dict) -> dict:
    issues: list[str] = []
    score = 100

    if not scene.survivors:
        issues.append("scene has no survivors")
        score -= 35
    if not scene.assets:
        issues.append("scene has no solid assets")
        score -= 25
    if not scene.hazards:
        issues.append("scene has no hazards")
        score -= 10
    if not env_scene.get("obstacles"):
        issues.append("converted environment has no obstacles")
        score -= 20
    if not env_scene.get("survivor_pos"):
        issues.append("converted environment has no active survivor target")
        score -= 30

    bounds = scene.bounds
    for survivor in scene.survivors:
        x, y, z = survivor.position
        if not (0 <= x <= bounds[0] and 0 <= y <= bounds[1] and z == 0):
            issues.append(f"survivor {survivor.profile.name} is outside navigable bounds")
            score -= 10
            break

    score = max(0, min(100, score))
    return {
        "score": score,
        "passed": score >= 60,
        "issues": issues,
        "feedback": (
            "Scene is valid for conversion and robot rollout."
            if score >= 60
            else "Scene needs repair before it is suitable for rollout."
        ),
    }


def _episode_response(result: dict, gif_url: str, max_steps: int) -> dict:
    return {
        "reached": bool(result.get("reached", False)),
        "fallen": bool(result.get("fallen", False)),
        "steps": int(result.get("steps", 0)),
        "total_reward": float(result.get("total_reward", 0.0)),
        "final_dist": result.get("final_dist"),
        "min_dist": result.get("min_dist"),
        "obstacle_contacts": result.get("obstacle_contacts", 0),
        "hazard_steps": result.get("hazard_steps", 0),
        "mean_stance_slip": result.get("mean_stance_slip", 0.0),
        "mean_assist_force": result.get("mean_assist_force", 0.0),
        "gait_score": result.get("gait_score", 0.0),
        "final_heading_error": result.get("final_heading_error"),
        "min_obstacle_clearance": result.get("min_obstacle_clearance"),
        "min_hazard_clearance": result.get("min_hazard_clearance"),
        "assist_scale": result.get("assist_scale", ASSIST_SCALE),
        "balance_assist_scale": result.get("balance_assist_scale", BALANCE_ASSIST_SCALE),
        "gif_url": gif_url,
        "trajectory": result.get("trajectory", []),
        "detection_event": result.get("detection_event"),
        "cancelled": bool(result.get("cancelled", False)),
        "completion_reason": result.get("completion_reason"),
        "frame_count": int(result.get("frame_count", 0)),
        "gif_fps": int(result.get("gif_fps", 20)),
        "gif_duration_seconds": float(result.get("gif_duration_seconds", 0.0)),
        "wall_time_seconds": result.get("wall_time_seconds"),
        "max_steps": max_steps,
    }


def _compose_scene_description(description: str, theme: str | None) -> str:
    trimmed_description = description.strip()
    trimmed_theme = (theme or "").strip()
    if not trimmed_theme:
        return trimmed_description

    lowered_description = trimmed_description.lower()
    lowered_theme = trimmed_theme.lower()
    if lowered_theme in lowered_description:
        return trimmed_description

    if trimmed_description:
        return f"{trimmed_theme}. {trimmed_description}"
    return trimmed_theme


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
        return {
            "ok": True,
            "model": MODEL_PATH,
            "max_steps": MAX_STEPS,
            "generated_max_steps": GENERATED_MAX_STEPS,
            "assist_scale": ASSIST_SCALE,
            "balance_assist_scale": BALANCE_ASSIST_SCALE,
        }

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

    @app.get("/runs/{run_id}/eval_live")
    def get_run_eval_live(run_id: str) -> dict:
        eval_live_path = run_dir(run_id) / "eval_live.json"
        if not eval_live_path.exists():
            raise HTTPException(status_code=404, detail=f"eval_live.json not found for run {run_id!r}")
        try:
            return read_json(eval_live_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/runs/{run_id}/episodes")
    def get_run_episodes(run_id: str) -> dict:
        eval_path = run_dir(run_id) / "eval.json"
        if not eval_path.exists():
            raise HTTPException(status_code=404, detail=f"eval.json not found for run {run_id!r}")
        try:
            eval_data = read_json(eval_path)
            return {
                "episodes": eval_data.get("scenes", []),
                "summary": {
                    "success_count": eval_data.get("success_count"),
                    "scene_count": eval_data.get("scene_count"),
                    "mean_reward": eval_data.get("mean_reward"),
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/runs/{run_id}/episodes/{scene_name}")
    def get_run_episode_detail(run_id: str, scene_name: str) -> dict:
        ep = load_episode(run_id, scene_name)
        if ep is None:
            raise HTTPException(
                status_code=404, detail=f"episode {scene_name!r} not found for run {run_id!r}"
            )
        return ep

    @app.post("/runs/{run_id}/episodes/{scene_name}/replay")
    def replay_episode(run_id: str, scene_name: str) -> dict:
        ep_dir = run_dir(run_id) / "episodes"
        if not ep_dir.exists():
            raise HTTPException(status_code=404, detail=f"episodes folder not found for run {run_id!r}")
        
        target_file = None
        for f in ep_dir.glob("*.json"):
            if f.stem == scene_name or f.stem.endswith(f"_{scene_name}"):
                target_file = f
                break
        if not target_file:
            raise HTTPException(
                status_code=404, detail=f"episode {scene_name!r} not found for run {run_id!r}"
            )

        gif_filename = f"replay_{target_file.stem}.gif"
        gif_path = run_dir(run_id) / gif_filename

        try:
            stats = replay_episode_actions(str(target_file), gif_path=str(gif_path))
            stats["gif_url"] = f"/run-gifs/{run_id}/{gif_filename}"
            return stats
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

    @app.post("/generate-prompt")
    def generate_prompt(req: GeneratePromptRequest) -> dict:
        survivor_count = 1
        profiles = normalize_survivor_profiles(survivor_count, None)
        try:
            prompt = generate_scene_prompt(
                difficulty=req.difficulty,
                survivor_count=survivor_count,
                theme=req.theme,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "prompt": prompt,
            "agent_prompt": build_generation_prompt(
                prompt,
                survivor_count,
                profiles,
                req.difficulty,
            ),
        }

    @app.post("/generate-scene")
    async def generate_scene(req: GenerateSceneRequest) -> dict:
        description = _compose_scene_description(req.description, req.theme)
        if not description:
            raise HTTPException(status_code=400, detail="description is required")

        survivor_count = 1
        max_steps = _episode_steps(req.max_steps, GENERATED_MAX_STEPS)

        try:
            scene = await asyncio.to_thread(
                ScenarioAgent().generate_scene,
                description,
                survivor_count,
                None,
                req.difficulty,
            )
            env_scene = scene_spec_to_env_scene(scene)
            scene_id = f"generated_{int(time.time())}_{uuid4().hex[:8]}"
            env_scene["name"] = scene_id
            eval_result = _score_generated_scene(scene, env_scene)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        episode = None
        episode_error = None
        if eval_result["passed"] and not req.skip_episode:
            gif_name = f"generated_{int(time.time() * 1000)}.gif"
            gif_path = _OUTPUT / gif_name
            key = f"generate:{gif_name}"
            cancel_event = _register_cancel(key)
            try:
                result = await asyncio.to_thread(
                    run_episode,
                    scene=env_scene,
                    model_path=MODEL_PATH,
                    gif_path=str(gif_path),
                    max_steps=max_steps,
                    assist_scale=ASSIST_SCALE,
                    balance_assist_scale=BALANCE_ASSIST_SCALE,
                    cancel_event=cancel_event,
                )
                episode = _episode_response(result, f"/gifs/{gif_name}", max_steps)
            except FileNotFoundError as e:
                episode_error = f"model not found: {e}"
            except Exception as e:
                episode_error = str(e)
            finally:
                _release_cancel(key, cancel_event)

        payload = {
            "scene_id": scene_id,
            "scene": scene.model_dump(),
            "env_scene": env_scene,
            "eval": eval_result,
            "episode": episode,
            "episode_error": episode_error,
            "default_max_steps": GENERATED_MAX_STEPS,
        }
        return _save_generated_session(payload)

    @app.get("/generated-scenes/{scene_id}")
    def get_generated_scene(scene_id: str) -> dict:
        return _load_generated_session(scene_id)

    @app.post("/generated-scenes/{scene_id}/run")
    async def run_generated_scene(scene_id: str, req: SceneRunRequest) -> dict:
        payload = _load_generated_session(scene_id)
        if not payload["eval"]["passed"]:
            raise HTTPException(status_code=409, detail="scene did not pass evaluation")

        max_steps = _episode_steps(req.max_steps, GENERATED_MAX_STEPS)
        gif_name = f"{scene_id}_{int(time.time() * 1000)}.gif"
        key = f"generate:{gif_name}"
        cancel_event = _register_cancel(key)
        try:
            result = await asyncio.to_thread(
                run_episode,
                scene=payload["env_scene"],
                model_path=MODEL_PATH,
                gif_path=str(_OUTPUT / gif_name),
                max_steps=max_steps,
                assist_scale=ASSIST_SCALE,
                balance_assist_scale=BALANCE_ASSIST_SCALE,
                cancel_event=cancel_event,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"model not found: {e}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            _release_cancel(key, cancel_event)

        payload["episode"] = _episode_response(result, f"/gifs/{gif_name}", max_steps)
        payload["episode_error"] = None
        return _save_generated_session(payload)

    @app.post("/scene/{idx}/run")
    async def run_scene(idx: int, req: SceneRunRequest) -> dict:
        if idx < 0 or idx >= len(GENERATED_SCENES):
            raise HTTPException(status_code=404, detail=f"scene {idx} not found")

        max_steps = _episode_steps(req.max_steps, MAX_STEPS)
        key = f"scene:{idx}"
        cancel_event = _register_cancel(key)
        try:
            result = await asyncio.to_thread(_run_scene, idx, max_steps, cancel_event)
            if not result.get("cancelled"):
                append_live_eval(current_run_id(), result)
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"model not found: {e}") from e
        finally:
            _release_cancel(key, cancel_event)

    @app.post("/command")
    async def command(req: CommandRequest) -> dict:
        decision = await asyncio.to_thread(get_gemini_target, req.text, SURVIVORS)
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
            "fallen": bool(result.get("fallen", False)),
            "steps": int(result.get("steps", 0)),
            "total_reward": float(result.get("total_reward", 0.0)),
            "final_dist": result.get("final_dist"),
            "min_dist": result.get("min_dist"),
            "obstacle_contacts": result.get("obstacle_contacts", 0),
            "hazard_steps": result.get("hazard_steps", 0),
            "mean_stance_slip": result.get("mean_stance_slip", 0.0),
            "mean_assist_force": result.get("mean_assist_force", 0.0),
            "gait_score": result.get("gait_score", 0.0),
            "assist_scale": result.get("assist_scale", ASSIST_SCALE),
            "balance_assist_scale": result.get("balance_assist_scale", BALANCE_ASSIST_SCALE),
            "gif_url": "/episode.gif",
            "trajectory": result.get("trajectory", []),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
