"""
Export a trained run into runs/<name>/ for demo + reproducibility.

Usage:
    uv run python export_run.py --model ./models/ppo_buried_detection_final
    uv run python export_run.py --model ./models/ppo_buried_detection_final --name my_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from disaster_env import DisasterEnv
from rescue_runner import run_generated_scene_suite
from scenes import GENERATED_SCENES
from run_store import (
    RUNS_DIR,
    MODELS_DIR,
    TB_LOGS_DIR,
    run_dir,
    read_json,
    write_json,
    probe_current_obs_dim,
)

console = Console()
_HERE = Path(__file__).resolve().parent


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _latest_tb_log() -> Path | None:
    if not TB_LOGS_DIR.exists():
        return None
    dirs = sorted(TB_LOGS_DIR.glob("PPO_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def extract_reward_curve(tb_log_dir: Path | None) -> list[dict]:
    """Pull rollout/ep_rew_mean from TensorBoard event files."""
    if tb_log_dir is None or not tb_log_dir.exists():
        return []

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    events = sorted(tb_log_dir.glob("events.out.tfevents.*"))
    if not events:
        return []

    acc = EventAccumulator(str(tb_log_dir), size_guidance={"scalars": 0})
    acc.Reload()

    tag = None
    for candidate in ("rollout/ep_rew_mean", "train/ep_rew_mean"):
        if candidate in acc.Tags().get("scalars", []):
            tag = candidate
            break
    if tag is None:
        return []

    return [
        {"step": int(e.step), "reward": round(float(e.value), 2)}
        for e in acc.Scalars(tag)
    ]


def _checkpoints_for_run(model_stem: str) -> list[str]:
    prefix = model_stem.replace("_final", "")
    found = sorted(MODELS_DIR.glob(f"{prefix}*.zip"))
    if not found and (MODELS_DIR / f"{model_stem}.zip").exists():
        found = [MODELS_DIR / f"{model_stem}.zip"]
    return [p.name for p in found]


def _build_advanced_stats(summary: dict, eval_data: dict) -> list[dict]:
    ev = eval_data
    stats = [
        {"group": "Run", "label": "Algorithm", "value": "PPO (Proximal Policy Optimization)"},
        {"group": "Run", "label": "Policy network", "value": summary.get("policy_network", "MLP 256 × 256")},
        {"group": "Run", "label": "Total timesteps", "value": f"{summary['total_steps']:,}"},
        {"group": "Run", "label": "Parallel envs", "value": str(summary["n_envs"])},
        {"group": "Run", "label": "Obs dimension", "value": str(summary["obs_dim"])},
        {"group": "Run", "label": "Exported", "value": summary["exported_at"][:10]},
        {"group": "Run", "label": "Git commit", "value": summary.get("git_commit") or "—"},
        {"group": "Hyperparameters", "label": "Batch size", "value": str(summary.get("batch_size", 256))},
        {"group": "Hyperparameters", "label": "n_steps", "value": str(summary.get("n_steps", 2048))},
        {"group": "Hyperparameters", "label": "Gamma (γ)", "value": str(summary.get("gamma", 0.99))},
        {"group": "Hyperparameters", "label": "GAE λ", "value": str(summary.get("gae_lambda", 0.95))},
        {"group": "Hyperparameters", "label": "Clip range", "value": str(summary.get("clip_range", 0.2))},
        {"group": "Evaluation", "label": "Scenes evaluated", "value": f"{ev['scene_count']} fixed disaster layouts"},
        {"group": "Evaluation", "label": "Success @ export", "value": f"{ev['success_count']} / {ev['scene_count']} scenes"},
        {"group": "Evaluation", "label": "Mean steps (reached)", "value": str(ev.get("mean_steps_reached", "—"))},
        {"group": "Evaluation", "label": "Mean reward", "value": str(ev.get("mean_reward", "—"))},
        {"group": "Evaluation", "label": "Buried detections", "value": str(ev.get("detection_count", 0))},
    ]
    return stats


def _curve_caption(curve: list[dict]) -> str:
    if len(curve) < 2:
        return "No TensorBoard curve exported — run training with tensorboard_log enabled."
    first, last = curve[0]["reward"], curve[-1]["reward"]
    delta = last - first
    direction = "improves" if delta > 0 else "declines"
    return (
        f"Mean episode reward from TensorBoard ({direction} {abs(delta):.0f} over "
        f"{curve[-1]['step']:,} steps)."
    )


def export_run(
    model_path: str | Path,
    run_name: str | None = None,
    *,
    max_steps: int = 300,
    tb_log_dir: Path | None = None,
    total_steps: int = 500_000,
    n_envs: int | None = None,
    no_rollout: bool = False,
    reached_only: bool = False,
    init_from: str | None = None,
    bc_transitions: int | None = None,
    bc_epochs: int | None = None,
) -> Path:
    model_path = Path(model_path)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    stem = model_path.name
    run_name = run_name or stem
    r_dir = run_dir(run_name)
    episodes_dir = r_dir / "episodes"
    gifs_dir = r_dir / "gifs"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    n_envs = n_envs or len(GENERATED_SCENES)
    tb_log_dir = Path(tb_log_dir).resolve() if tb_log_dir else _latest_tb_log()
    curve = extract_reward_curve(tb_log_dir)

    # Probe obs dim from env
    obs_dim = probe_current_obs_dim()

    console.print(f"[bold cyan]Evaluating[/] {stem} across {len(GENERATED_SCENES)} scenes…")
    suite_results = run_generated_scene_suite(
        model_path=str(model_path),
        output_dir=str(gifs_dir),
        max_steps=max_steps,
        record_rollout=True,
    )

    episode_exports = []
    for r in suite_results:
        idx = r["scene_index"]
        name = r["scene_name"]
        ep_file = f"{idx + 1:02d}_{name}.json"
        gif_name = f"{idx + 1:02d}_{name}.gif"
        
        ep_payload = {
            "scene_index": idx,
            "scene_name": name,
            "reached": r["reached"],
            "steps": r["steps"],
            "total_reward": r["total_reward"],
            "trajectory": r["trajectory"],
            "detection_event": r.get("detection_event"),
            "gif": f"gifs/{gif_name}",
            "model": stem,
        }
        if not no_rollout and r.get("rollout"):
            ep_payload["rollout"] = r["rollout"]

        has_ep_file = not (reached_only and not r["reached"])
        if has_ep_file:
            (episodes_dir / ep_file).write_text(json.dumps(ep_payload, indent=2))

        episode_exports.append(
            {
                "scene_index": idx,
                "scene_name": name,
                "reached": r["reached"],
                "steps": r["steps"],
                "total_reward": r["total_reward"],
                "detection_event": r.get("detection_event") is not None,
                "episode_file": f"episodes/{ep_file}" if has_ep_file else None,
                "gif": f"gifs/{gif_name}",
            }
        )

    reached_results = [r for r in suite_results if r["reached"]]
    success_count = len(reached_results)
    eval_data = {
        "model": stem,
        "max_steps": max_steps,
        "scene_count": len(suite_results),
        "success_count": success_count,
        "mean_steps_reached": (
            round(sum(r["steps"] for r in reached_results) / len(reached_results), 1)
            if reached_results
            else None
        ),
        "mean_reward": round(
            sum(r["total_reward"] for r in suite_results) / len(suite_results), 2
        ),
        "detection_count": sum(1 for r in suite_results if r.get("detection_event")),
        "scenes": episode_exports,
    }

    exported_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "id": run_name,
        "name": run_name,
        "model_path": f"models/{stem}.zip",
        "total_steps": total_steps,
        "n_envs": n_envs,
        "obs_dim": obs_dim,
        "max_steps_eval": max_steps,
        "policy_network": "MLP 256 × 256",
        "batch_size": 256,
        "n_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "tensorboard_log": (
            str(tb_log_dir.relative_to(_HERE))
            if tb_log_dir and str(tb_log_dir).startswith(str(_HERE))
            else str(tb_log_dir) if tb_log_dir else None
        ),
        "git_commit": _git_commit(),
        "exported_at": exported_at,
        "checkpoints": _checkpoints_for_run(stem),
    }
    if init_from is not None:
        summary["init_from"] = init_from
    if bc_transitions is not None:
        summary["bc_transitions"] = bc_transitions
    if bc_epochs is not None:
        summary["bc_epochs"] = bc_epochs

    y_values = [p["reward"] for p in curve] if curve else [-250, 50]
    y_min = min(y_values) - 20
    y_max = max(y_values) + 20

    manifest = {
        **summary,
        "subtitle": f"{n_envs}-scene vectorized PPO · obs {obs_dim}D",
        "solved_threshold": 0,
        "y_min": round(y_min, 1),
        "y_max": round(y_max, 1),
        "caption": _curve_caption(curve),
        "curve": curve,
        "eval": eval_data,
        "file_tree": [
            f"runs/{run_name}/",
            "  ├── summary.json",
            "  ├── curve.json",
            "  ├── eval.json",
            "  ├── episodes/          # trajectories + obs/action rollouts",
            "  └── gifs/              # eval GIFs per scene",
        ],
        "advanced_stats": _build_advanced_stats(summary, eval_data),
    }

    write_json(r_dir / "summary.json", summary)
    write_json(r_dir / "curve.json", curve)
    write_json(r_dir / "eval.json", eval_data)
    write_json(r_dir / "manifest.json", manifest)

    table = Table(title=f"Exported → runs/{run_name}/")
    table.add_column("Scene")
    table.add_column("Reached")
    table.add_column("Steps")
    table.add_column("Detected")
    for ep in episode_exports:
        table.add_row(
            ep["scene_name"],
            "Yes" if ep["reached"] else "No",
            str(ep["steps"]),
            "Yes" if ep["detection_event"] else "—",
        )
    console.print(table)
    console.print(
        f"[bold green]✓[/] {success_count}/{len(suite_results)} scenes · "
        f"{len(curve)} curve points · runs/{run_name}/"
    )
    return r_dir


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            m = read_json(manifest_path)
        else:
            summary_path = d / "summary.json"
            if not summary_path.exists():
                continue
            m = read_json(summary_path)
        runs.append(
            {
                "id": m.get("id", d.name),
                "name": m.get("name", d.name),
                "subtitle": m.get("subtitle", ""),
                "total_steps": m.get("total_steps", 0),
                "success_count": m.get("eval", {}).get("success_count"),
                "scene_count": m.get("eval", {}).get("scene_count"),
                "exported_at": m.get("exported_at"),
            }
        )
    return runs


def load_run_manifest(run_id: str) -> dict | None:
    manifest_path = run_dir(run_id) / "manifest.json"
    if not manifest_path.exists():
        return None
    return read_json(manifest_path)


def manifest_to_frontend(m: dict) -> dict:
    """Shape expected by TrainingRuns.tsx."""
    eval_live_path = run_dir(m["id"]) / "eval_live.json"
    eval_live_data = None
    if eval_live_path.exists():
        try:
            el = read_json(eval_live_path)
            sessions = el.get("sessions", [])
            if sessions:
                eval_live_data = sessions[-1]
        except Exception:
            pass

    res = {
        "id": m["id"],
        "name": m["name"],
        "subtitle": m.get("subtitle", ""),
        "totalSteps": m.get("total_steps", 0),
        "solvedThreshold": m.get("solved_threshold", 0),
        "yMin": m.get("y_min", -250),
        "yMax": m.get("y_max", 50),
        "caption": m.get("caption", ""),
        "curve": m.get("curve", []),
        "checkpoints": m.get("checkpoints", []),
        "fileTree": m.get("file_tree", []),
        "advancedStats": m.get("advanced_stats", []),
        "eval": m.get("eval"),
        "exportedAt": m.get("exported_at"),
        "git_commit": m.get("git_commit"),
        "source": "runs",
    }
    if eval_live_data is not None:
        res["evalLive"] = eval_live_data
    return res


def main():
    parser = argparse.ArgumentParser(description="Export a training run to runs/<name>/")
    parser.add_argument("--model", default="./models/ppo_buried_detection_final")
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--tb-log", default=None, help="TensorBoard log dir (default: latest PPO_*)")
    parser.add_argument("--no-rollout", action="store_true", help="skip obs/action arrays (smaller files)")
    parser.add_argument("--reached-only", action="store_true", help="only write episodes where reached: true")
    args = parser.parse_args()

    tb = Path(args.tb_log) if args.tb_log else None
    export_run(
        args.model,
        args.name,
        max_steps=args.max_steps,
        tb_log_dir=tb,
        total_steps=args.total_steps,
        no_rollout=args.no_rollout,
        reached_only=args.reached_only,
    )


if __name__ == "__main__":
    main()
