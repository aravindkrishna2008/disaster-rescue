from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import mujoco
import mujoco.viewer

from disaster_env import DisasterEnv
from scenario_agent import SceneSpec
from scene_adapter import scene_spec_to_env_scene


DEFAULT_SCENE_JSON = Path("tmp/generated_scene.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a ScenarioAgent-generated JSON scene in DisasterEnv."
    )
    parser.add_argument(
        "scene_json",
        nargs="?",
        default=DEFAULT_SCENE_JSON,
        help=f"Path to generated scene JSON. Defaults to {DEFAULT_SCENE_JSON}.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Render one offscreen RGB frame instead of opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=18.0,
        help="Initial viewer camera distance. Increase to zoom out. Defaults to 18.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=45.0,
        help="Initial viewer camera azimuth in degrees. Defaults to 45.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-55.0,
        help="Initial viewer camera elevation in degrees. Defaults to -55.",
    )
    args = parser.parse_args()

    scene_path = Path(args.scene_json)
    scene = SceneSpec.model_validate_json(scene_path.read_text(encoding="utf-8"))
    env_scene = scene_spec_to_env_scene(scene)
    render_mode = "rgb_array"

    env = DisasterEnv(scene=env_scene, render_mode=render_mode)
    try:
        obs, _ = env.reset()
        active = env_scene["active_survivor"]
        print(f"Loaded: {scene_path}")
        print(f"Active survivor: {active['name']} ({active['type']}, {active['priority']})")
        print(f"Observation shape: {obs.shape}")
        if args.rgb:
            frame = env.render()
            print(f"Rendered frame shape: {None if frame is None else frame.shape}")
        else:
            open_passive_viewer(env, args.distance, args.azimuth, args.elevation)
            print("Close the MuJoCo viewer window to exit.")
    finally:
        env.close()


def open_passive_viewer(
    env: DisasterEnv,
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    try:
        viewer_context = mujoco.viewer.launch_passive(env._model, env._data)
    except RuntimeError as exc:
        if "mjpython" not in str(exc):
            raise
        script = Path(__file__).name
        args = " ".join(sys.argv[1:])
        command = f"uv run mjpython {script}"
        if args:
            command = f"{command} {args}"
        raise SystemExit(
            "MuJoCo's interactive viewer requires mjpython on macOS.\n"
            f"Run this instead:\n  {command}"
        ) from None

    with viewer_context as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation

        print("Viewer controls: right-drag/scroll to zoom, left-drag to rotate, shift-drag to pan.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
