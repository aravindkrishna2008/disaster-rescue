"""
rescue_runner.py — Person C calls run_episode(); gets back stats + GIF path.

Usage:
    from rescue_runner import run_episode
    result = run_episode(scene, model_path="./models/ppo_model_final")

Returns:
    {
        "trajectory": [[x, y, z], ...],
        "reached": bool,
        "steps": int,
        "gif_path": str,        # e.g. "./output/episode.gif"
        "total_reward": float,
    }
"""

import os
import numpy as np
import imageio
from stable_baselines3 import PPO

from disaster_env import DisasterEnv, DEFAULT_SCENE
from scenes import GENERATED_SCENES, get_scene

OUTPUT_DIR   = "./output"
GIF_FPS      = 20
GIF_FILENAME = "episode.gif"


def run_episode(
    scene: dict = None,
    model_path: str = "./models/ppo_model_final",
    gif_path: str = None,
    max_steps: int = 600,
) -> dict:
    """
    Run one episode of the trained policy in the given scene.

    Args:
        scene:      Scene config dict (from Gemini). Falls back to DEFAULT_SCENE.
        model_path: Path to the SB3 PPO model (no .zip extension needed).
        gif_path:   Where to save the GIF. Defaults to ./output/episode.gif
        max_steps:  Hard cap on episode length.

    Returns:
        dict with trajectory, reached, steps, gif_path, total_reward.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if gif_path is None:
        gif_path = os.path.join(OUTPUT_DIR, GIF_FILENAME)

    scene = scene or DEFAULT_SCENE

    env = DisasterEnv(scene=scene, render_mode="rgb_array")
    obs, _ = env.reset()

    model = PPO.load(model_path, device="cpu")

    trajectory: list[list[float]] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    reached = False

    for _ in range(max_steps):
        # Capture frame before step (so first frame shows start pose)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        robot_pos = [float(obs[0]), float(obs[1]), 0.0]
        trajectory.append(robot_pos)

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
        "steps": len(trajectory),
        "gif_path": gif_path,
        "total_reward": round(total_reward, 2),
    }


def run_generated_scene_suite(
    model_path: str = "./models/ppo_fixed_six_final",
    output_dir: str = OUTPUT_DIR,
    max_steps: int = 600,
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
        )
        result["scene_index"] = idx
        result["scene_name"] = name
        results.append(result)
    return results


# ── quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table

    console = Console()

    model_arg = sys.argv[1] if len(sys.argv) > 1 else "./models/ppo_fixed_six_final"
    suite = "--suite" in sys.argv

    console.print(f"[bold cyan]Running episode with model:[/] {model_arg}")
    if suite:
        table = Table(title="Generated Scene Suite", show_header=True)
        table.add_column("#", style="bold")
        table.add_column("Scene")
        table.add_column("Reached")
        table.add_column("Steps")
        table.add_column("Reward")
        table.add_column("GIF")

        for result in run_generated_scene_suite(model_path=model_arg):
            table.add_row(
                str(result["scene_index"] + 1),
                result["scene_name"],
                "Yes" if result["reached"] else "No",
                str(result["steps"]),
                str(result["total_reward"]),
                result["gif_path"],
            )

        console.print(table)
        raise SystemExit(0)

    result = run_episode(model_path=model_arg)

    table = Table(title="Episode Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Steps",         str(result["steps"]))
    table.add_row("Survivor Reached", "✓ Yes" if result["reached"] else "✗ No")
    table.add_row("Total Reward",  str(result["total_reward"]))
    table.add_row("GIF saved to",  result["gif_path"])
    table.add_row("Trajectory pts", str(len(result["trajectory"])))

    console.print(table)
