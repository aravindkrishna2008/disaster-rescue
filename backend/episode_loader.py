"""
episode_loader.py — Utility functions to load exported episodes, extract transitions
for Behavior Cloning, and replay episodes in the DisasterEnv simulator.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from disaster_env import DisasterEnv
from scenes import get_scene
from run_store import run_dir, read_json


def load_episodes(run_id: str, reached_only: bool = False) -> list[dict]:
    """Load all episode JSON files for a given run."""
    ep_dir = run_dir(run_id) / "episodes"
    if not ep_dir.exists():
        return []

    episodes = []
    for f in sorted(ep_dir.glob("*.json")):
        try:
            ep = read_json(f)
            if reached_only and not ep.get("reached", False):
                continue
            episodes.append(ep)
        except Exception:
            pass
    return episodes


def load_transitions(run_id: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flatten and return all rollout transitions (obs, action) from reached episodes."""
    episodes = load_episodes(run_id, reached_only=True)
    transitions = []

    for ep in episodes:
        rollout = ep.get("rollout")
        if not rollout:
            continue
        for step in rollout:
            if "obs" in step and "action" in step:
                obs = np.array(step["obs"], dtype=np.float32)
                act = np.array(step["action"], dtype=np.float32)
                transitions.append((obs, act))

    return transitions


def replay_episode_actions(episode_path: str | Path, gif_path: str | Path | None = None) -> dict:
    """
    Reset DisasterEnv for the episode's scene, step using saved rollout actions,
    and compute a trajectory match score against the saved trajectory.
    """
    with open(episode_path, "r", encoding="utf-8") as f:
        ep = json.load(f)

    rollout = ep.get("rollout")
    if not rollout:
        raise ValueError("Episode has no rollout data to replay")

    scene_index = ep["scene_index"]
    scene = get_scene(scene_index)

    # Initialize environment
    env = DisasterEnv(scene=scene, render_mode="rgb_array")
    
    # Check observation dimension compatibility
    saved_obs_dim = len(rollout[0]["obs"])
    current_obs_dim = env.observation_space.shape[0]
    if saved_obs_dim != current_obs_dim:
        env.close()
        raise ValueError(
            f"Observation dimension mismatch: saved episode has {saved_obs_dim}D, "
            f"but current environment expects {current_obs_dim}D"
        )

    frames = []
    replayed_xy = []
    reached = False

    try:
        env.reset()
        
        # Capture starting frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        for step in rollout:
            action = np.array(step["action"], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record xy position of the robot from the observation
            replayed_xy.append((float(obs[0]), float(obs[1])))

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                reached = info.get("reached", False)
                break
    finally:
        env.close()

    # Save GIF if requested
    if gif_path and frames:
        import imageio
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(gif_path), frames, fps=20, loop=0)

    # Calculate mean Euclidean distance trajectory match score
    saved_trajectory = ep.get("trajectory", [])
    n_compare = min(len(replayed_xy), len(saved_trajectory))
    if n_compare > 0:
        distances = []
        for i in range(n_compare):
            pt_replayed = np.array(replayed_xy[i])
            pt_saved = np.array(saved_trajectory[i][:2])
            distances.append(np.linalg.norm(pt_replayed - pt_saved))
        trajectory_match_score = float(np.mean(distances))
    else:
        trajectory_match_score = 0.0

    return {
        "reached": reached,
        "steps": len(replayed_xy),
        "trajectory_match_score": round(trajectory_match_score, 4),
    }
