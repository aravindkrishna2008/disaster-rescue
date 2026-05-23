"""
run_store.py — Shared utilities for run directories, JSON files, live evaluation,
and episode rollouts.
"""

from __future__ import annotations

import os
import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from disaster_env import DisasterEnv
from scenes import GENERATED_SCENES

_HERE = Path(__file__).resolve().parent
RUNS_DIR = _HERE / "runs"
MODELS_DIR = _HERE / "models"
TB_LOGS_DIR = _HERE / "tb_logs"

_eval_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    """Return path to a specific run directory."""
    return RUNS_DIR / run_id


def read_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    """Write dictionary to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def current_run_id() -> str:
    """Derive active run name from BATTLE_ANGEL_MODEL environment variable or fallback."""
    model_path = os.environ.get(
        "BATTLE_ANGEL_MODEL", str(MODELS_DIR / "ppo_buried_detection_final")
    )
    return Path(model_path).stem


def append_live_eval(run_id: str, scene_result: dict) -> None:
    """Atomic read-modify-write of runs/<run_id>/eval_live.json."""
    path = run_dir(run_id) / "eval_live.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with _eval_lock:
        if path.exists():
            try:
                data = read_json(path)
            except Exception:
                data = {}
        else:
            data = {}

        data.setdefault("run_id", run_id)

        # Derive model path from summary.json if available, or default
        summary_path = run_dir(run_id) / "summary.json"
        model_path = f"models/{run_id}.zip"
        if summary_path.exists():
            try:
                s_data = read_json(summary_path)
                model_path = s_data.get("model_path", model_path)
            except Exception:
                pass
        data.setdefault("model_path", model_path)

        now_str = datetime.now(timezone.utc).isoformat()
        data["updated_at"] = now_str

        new_scene = {
            "scene_index": scene_result["scene_index"],
            "scene_name": scene_result["scene_name"],
            "reached": bool(scene_result["reached"]),
            "steps": int(scene_result["steps"]),
            "total_reward": round(float(scene_result["total_reward"]), 2),
            "detection_event": bool(scene_result.get("detection_event")),
            "gif_url": scene_result.get("gif_url"),
        }

        new_session = {
            "timestamp": now_str,
            "source": "mission_control",
            "max_steps": scene_result.get("max_steps", 300),
            "scenes": [new_scene]
        }

        sessions = data.setdefault("sessions", [])
        sessions.append(new_session)

        # Atomic write using a temp file
        fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix="eval_live_tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, str(path))
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise e


def load_episode(run_id: str, scene_key: str) -> dict | None:
    """Read episodes/{file}.json for a run, matching by full filename or suffix/name."""
    ep_dir = run_dir(run_id) / "episodes"
    if not ep_dir.exists():
        return None

    # 1. Try direct exact match
    p = ep_dir / scene_key
    if p.exists() and p.is_file():
        return read_json(p)
    p_json = ep_dir / f"{scene_key}.json"
    if p_json.exists() and p_json.is_file():
        return read_json(p_json)

    # 2. Try prefix/suffix matching (e.g., match 'downtown_rubble' to '06_downtown_rubble.json')
    for f in ep_dir.glob("*.json"):
        if f.stem == scene_key or f.stem.endswith(f"_{scene_key}"):
            return read_json(f)

    return None


def probe_current_obs_dim() -> int:
    """Run a single-step DisasterEnv probe to determine the current observation dimension."""
    probe_env = DisasterEnv(scene=GENERATED_SCENES[0], render_mode="rgb_array")
    obs_dim = int(probe_env.observation_space.shape[0])
    probe_env.close()
    return obs_dim
