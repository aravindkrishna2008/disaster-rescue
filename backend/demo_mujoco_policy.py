"""Run the trained G1 rescue policy in a live MuJoCo viewer."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from disaster_env import DEFAULT_BALANCE_ASSIST_SCALE, DEFAULT_SCENE, DisasterEnv
from scenes import get_scene


STRESS_SCENE = {
    "name": "multi_level_rubble_stress",
    "robot_start": [-5.5, -5.0, 1.0],
    "survivor_pos": [5.0, 4.5, 1.0],
    "obstacles": [
        {"pos": [-2.5, -1.5, 1.0], "size": [0.35, 2.0, 1.0]},
        {"pos": [0.8, 1.2, 0.8], "size": [0.4, 1.6, 0.8]},
        {"pos": [3.0, -2.3, 1.1], "size": [0.5, 0.8, 1.1]},
        {"pos": [-6.6, -1.1, 0.75], "size": [0.35, 0.85, 0.75]},
        {"pos": [-6.3, 3.5, 0.75], "size": [0.75, 0.35, 0.75]},
        {"pos": [1.4, -5.55, 0.75], "size": [0.85, 0.35, 0.75]},
        {"pos": [5.8, -0.9, 0.75], "size": [0.35, 0.85, 0.75]},
        {"pos": [6.0, 2.6, 0.7], "size": [0.75, 0.35, 0.7]},
    ],
    "terrain": [
        {"pos": [-5.0, -4.25, 0.02], "size": [0.55, 0.5, 0.02], "rgba": "0.26 0.29 0.25 1"},
        {"pos": [-4.25, -3.5, 0.028], "size": [0.55, 0.5, 0.028], "rgba": "0.33 0.34 0.29 1"},
        {"pos": [-4.0, -2.75, 0.022], "size": [0.55, 0.48, 0.022], "euler": [0.0, 0.05, -0.08]},
        {"pos": [-4.0, -1.25, 0.035], "size": [0.55, 0.48, 0.035], "rgba": "0.29 0.32 0.30 1"},
        {"pos": [-3.8, 0.35, 0.025], "size": [0.55, 0.48, 0.025], "euler": [0.0, -0.04, 0.1]},
        {"pos": [-3.15, 1.65, 0.04], "size": [0.6, 0.48, 0.04], "rgba": "0.38 0.36 0.31 1"},
        {"pos": [-2.1, 2.25, 0.028], "size": [0.55, 0.48, 0.028], "euler": [0.0, 0.05, -0.08]},
        {"pos": [-0.75, 2.6, 0.045], "size": [0.6, 0.5, 0.045], "rgba": "0.31 0.34 0.31 1"},
        {"pos": [0.8, 3.0, 0.03], "size": [0.6, 0.5, 0.03], "euler": [0.0, -0.04, 0.09]},
        {"pos": [2.0, 3.45, 0.025], "size": [0.55, 0.48, 0.025], "rgba": "0.27 0.30 0.28 1"},
        {"pos": [3.1, 4.0, 0.038], "size": [0.6, 0.5, 0.038], "rgba": "0.35 0.35 0.30 1"},
        {"pos": [4.25, 4.35, 0.026], "size": [0.55, 0.45, 0.026], "euler": [0.0, 0.04, -0.08]},
    ],
    "hazards": [
        {"center": [-0.5, -3.0, 0.0], "radius": 0.9},
        {"center": [2.2, 2.7, 0.0], "radius": 1.0},
        {"center": [-3.8, 1.5, 0.0], "radius": 0.7},
    ],
    "difficulty": "stress",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a live MuJoCo policy demo.")
    parser.add_argument("--model", default="./models/g1_locomotion_natural_final")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--stress-scene", action="store_true")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--assist-scale", type=float, default=0.95)
    parser.add_argument("--balance-assist-scale", type=float, default=DEFAULT_BALANCE_ASSIST_SCALE)
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = STRESS_SCENE if args.stress_scene else get_scene(args.scene_index) if args.scene_index >= 0 else DEFAULT_SCENE
    env = DisasterEnv(
        scene=scene,
        render_mode="rgb_array",
        curriculum_stage="natural_target",
        assist_scale=args.assist_scale,
        balance_assist_scale=args.balance_assist_scale,
    )
    model = PPO.load(args.model, device="cpu")
    obs, _ = env.reset()

    timestep = float(env._model.opt.timestep) * 10.0
    sleep_s = max(timestep / max(args.speed, 0.1), 0.001)

    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env._pelvis_body_id
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 135.0

        for _ in range(args.max_steps):
            if not viewer.is_running():
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()

            if terminated or truncated:
                print(
                    "Episode finished:",
                    {
                        "reached": bool(info["reached"]),
                        "fallen": bool(info["fallen"]),
                        "steps": int(info["steps"]),
                        "dist": round(float(info["dist"]), 3),
                        "obstacle_contacts": int(info["obstacle_contacts"]),
                        "hazard_steps": int(info["hazard_steps"]),
                        "mean_stance_slip": round(float(info["mean_stance_slip"]), 3),
                        "gait_score": round(float(info["gait_score"]), 3),
                        "assist_scale": round(float(info["assist_scale"]), 3),
                        "balance_assist_scale": round(float(info["balance_assist_scale"]), 3),
                    },
                )
                break

            time.sleep(sleep_s)

    env.close()


if __name__ == "__main__":
    main()
