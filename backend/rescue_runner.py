"""
rescue_runner.py — Person C calls run_episode(); gets back stats + GIF path.

Usage:
    from rescue_runner import run_episode
    result = run_episode(scene, model_path="./models/g1_locomotion_walk_final")

Returns:
    {
        "trajectory": [[x, y, z], ...],
        "reached": bool,
        "fallen": bool,
        "steps": int,
        "gif_path": str,        # e.g. "./output/episode.gif"
        "total_reward": float,
        "final_dist": float,
        "min_dist": float,
    }
"""

import argparse
import os
import numpy as np
import imageio
from stable_baselines3 import PPO

from disaster_env import DEFAULT_BALANCE_ASSIST_SCALE, DisasterEnv, DEFAULT_SCENE
from scenes import GENERATED_SCENES, get_scene

OUTPUT_DIR   = "./output"
GIF_FPS      = 20
GIF_FILENAME = "episode.gif"
DEFAULT_MODEL_PATH = "./models/g1_locomotion_natural_final"


def run_episode(
    scene: dict = None,
    model_path: str = DEFAULT_MODEL_PATH,
    gif_path: str = None,
    max_steps: int = 1_000,
    curriculum_stage: str = "natural_target",
    assist_scale: float = 0.95,
    balance_assist_scale: float = DEFAULT_BALANCE_ASSIST_SCALE,
) -> dict:
    """
    Run one episode of the trained policy in the given scene.

    Args:
        scene:      Scene config dict (from Gemini). Falls back to DEFAULT_SCENE.
        model_path: Path to the SB3 PPO model (no .zip extension needed).
        gif_path:   Where to save the GIF. Defaults to ./output/episode.gif
        max_steps:  Hard cap on episode length.

    Returns:
        dict with trajectory, reached/fallen flags, distance metrics, gif_path, and reward.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if gif_path is None:
        gif_path = os.path.join(OUTPUT_DIR, GIF_FILENAME)

    scene = scene or DEFAULT_SCENE

    env = DisasterEnv(
        scene=scene,
        render_mode="rgb_array",
        curriculum_stage=curriculum_stage,
        assist_scale=assist_scale,
        balance_assist_scale=balance_assist_scale,
    )
    obs, _ = env.reset()

    model = PPO.load(model_path, device="cpu")

    trajectory: list[list[float]] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    reached = False
    fallen = False
    final_dist = None
    min_dist = None
    final_heading_error = None
    final_info = {}

    for _ in range(max_steps):
        # Capture frame before step (so first frame shows start pose)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        base_pos = info.get("base_pos", obs[:3])
        trajectory.append([float(base_pos[0]), float(base_pos[1]), float(base_pos[2])])
        dist = float(info.get("dist", np.inf))
        final_dist = dist
        min_dist = dist if min_dist is None else min(min_dist, dist)
        final_heading_error = info.get("heading_error")
        fallen = bool(info.get("fallen", False))
        final_info = info

        if terminated or truncated:
            reached = info.get("reached", False)
            break

    # Capture final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    env.close()

    # Save GIF
    if frames:
        imageio.mimsave(gif_path, frames, fps=GIF_FPS, loop=0)

    return {
        "trajectory": trajectory,
        "reached": reached,
        "fallen": fallen,
        "steps": len(trajectory),
        "gif_path": gif_path,
        "total_reward": round(total_reward, 2),
        "final_dist": None if final_dist is None else round(float(final_dist), 3),
        "min_dist": None if min_dist is None else round(float(min_dist), 3),
        "final_heading_error": (
            None if final_heading_error is None else round(float(final_heading_error), 3)
        ),
        "obstacle_contacts": int(final_info.get("obstacle_contacts", 0)),
        "hazard_steps": int(final_info.get("hazard_steps", 0)),
        "min_obstacle_clearance": round(float(final_info.get("min_obstacle_clearance", 0.0)), 3),
        "min_hazard_clearance": round(float(final_info.get("min_hazard_clearance", 0.0)), 3),
        "mean_stance_slip": round(float(final_info.get("mean_stance_slip", 0.0)), 3),
        "mean_assist_force": round(float(final_info.get("mean_assist_force_episode", 0.0)), 3),
        "gait_score": round(float(final_info.get("gait_score", 0.0)), 3),
        "assist_scale": round(float(final_info.get("assist_scale", assist_scale)), 3),
        "balance_assist_scale": round(float(final_info.get("balance_assist_scale", balance_assist_scale)), 3),
    }


def run_generated_scene_suite(
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = OUTPUT_DIR,
    max_steps: int = 1_000,
    assist_scale: float = 0.95,
    balance_assist_scale: float = DEFAULT_BALANCE_ASSIST_SCALE,
) -> list[dict]:
    """Render one deterministic PPO episode for each fixed generated scene."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for idx, scene_template in enumerate(GENERATED_SCENES):
        scene = get_scene(idx)
        name = scene_template["name"]
        gif_path = os.path.join(output_dir, f"{idx + 1:02d}_{name}.gif")
        result = run_episode(
            scene=scene,
            model_path=model_path,
            gif_path=gif_path,
            max_steps=max_steps,
            assist_scale=assist_scale,
            balance_assist_scale=balance_assist_scale,
        )
        result["scene_index"] = idx
        result["scene_name"] = name
        results.append(result)
    return results


# ── quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Run deterministic G1 rescue rollouts.")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--assist-scale", type=float, default=0.95)
    parser.add_argument("--balance-assist-scale", type=float, default=DEFAULT_BALANCE_ASSIST_SCALE)
    args = parser.parse_args()

    console = Console()

    console.print(f"[bold cyan]Running episode with model:[/] {args.model}")
    if args.suite:
        table = Table(title="Generated Scene Suite", show_header=True)
        table.add_column("#", style="bold")
        table.add_column("Scene")
        table.add_column("Reached")
        table.add_column("Fallen")
        table.add_column("Final Dist")
        table.add_column("Min Dist")
        table.add_column("Obs C")
        table.add_column("Haz Steps")
        table.add_column("Slip")
        table.add_column("Assist")
        table.add_column("Balance")
        table.add_column("Gait")
        table.add_column("Steps")
        table.add_column("Reward")
        table.add_column("GIF")

        for result in run_generated_scene_suite(
            model_path=args.model,
            max_steps=args.max_steps,
            assist_scale=args.assist_scale,
            balance_assist_scale=args.balance_assist_scale,
        ):
            table.add_row(
                str(result["scene_index"] + 1),
                result["scene_name"],
                "Yes" if result["reached"] else "No",
                "Yes" if result["fallen"] else "No",
                str(result["final_dist"]),
                str(result["min_dist"]),
                str(result["obstacle_contacts"]),
                str(result["hazard_steps"]),
                str(result["mean_stance_slip"]),
                str(result["mean_assist_force"]),
                str(result["balance_assist_scale"]),
                str(result["gait_score"]),
                str(result["steps"]),
                str(result["total_reward"]),
                result["gif_path"],
            )

        console.print(table)
        raise SystemExit(0)

    result = run_episode(
        model_path=args.model,
        max_steps=args.max_steps,
        assist_scale=args.assist_scale,
        balance_assist_scale=args.balance_assist_scale,
    )

    table = Table(title="Episode Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Steps",         str(result["steps"]))
    table.add_row("Survivor Reached", "Yes" if result["reached"] else "No")
    table.add_row("Fallen", "Yes" if result["fallen"] else "No")
    table.add_row("Final Dist", str(result["final_dist"]))
    table.add_row("Min Dist", str(result["min_dist"]))
    table.add_row("Final Heading Error", str(result["final_heading_error"]))
    table.add_row("Obstacle Contacts", str(result["obstacle_contacts"]))
    table.add_row("Hazard Steps", str(result["hazard_steps"]))
    table.add_row("Min Obstacle Clearance", str(result["min_obstacle_clearance"]))
    table.add_row("Min Hazard Clearance", str(result["min_hazard_clearance"]))
    table.add_row("Mean Stance Slip", str(result["mean_stance_slip"]))
    table.add_row("Mean Assist Force", str(result["mean_assist_force"]))
    table.add_row("Gait Score", str(result["gait_score"]))
    table.add_row("Assist Scale", str(result["assist_scale"]))
    table.add_row("Balance Assist Scale", str(result["balance_assist_scale"]))
    table.add_row("Total Reward",  str(result["total_reward"]))
    table.add_row("GIF saved to",  result["gif_path"])
    table.add_row("Trajectory pts", str(len(result["trajectory"])))

    console.print(table)
