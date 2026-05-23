from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from disaster_env import DisasterEnv

from scenario_agent import ScenarioAgent, SurvivorProfile
from scene_adapter import scene_spec_to_env_scene

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_ROOT / "tmp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test ScenarioAgent.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompts, raw agent output, and repaired scene JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for the managed-agent interaction. Defaults to 45.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Also write an HTML preview for debugging. Scene JSON is always updated.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline using a mock generated scene instead of contacting Gemini.",
    )
    args = parser.parse_args()

    TEMP_DIR.mkdir(exist_ok=True)

    if args.mock:
        from scenario_agent import (
            ASSET_CATALOG,
            HazardZone,
            PlacedAsset,
            SceneSpec,
            SurvivorLocation,
        )

        scene = SceneSpec(
            description="Mock collapsed building",
            difficulty="easy",
            robot_start=(1.5, 1.5, 0.0),
            survivors=[
                SurvivorLocation(
                    profile=SurvivorProfile(
                        name="Maya", type="baby", priority="critical"
                    ),
                    position=(4.0, 16.0, 0.0),
                ),
                SurvivorLocation(
                    profile=SurvivorProfile(
                        name="Luis", type="adult", priority="medium"
                    ),
                    position=(18.0, 18.0, 0.0),
                ),
            ],
            assets=[
                PlacedAsset(
                    asset_id="concrete_slab",
                    position=(10.0, 10.0, ASSET_CATALOG["concrete_slab"].size[2] / 2),
                    size=ASSET_CATALOG["concrete_slab"].size,
                    rotation_yaw=27.0,
                )
            ],
            hazards=[
                HazardZone(
                    hazard_id="gas-1",
                    type="gas",
                    center=(5.0, 5.0, 0.0),
                    radius=1.25,
                    severity="high",
                )
            ],
        )
    else:
        agent = ScenarioAgent()
        agent.create()

        scene = agent.generate_scene(
            "Collapsed apartment after earthquake with no fire hazards.",
            survivor_count=2,
            survivor_profiles=[
                SurvivorProfile(name="Maya", type="baby", priority="critical"),
                SurvivorProfile(name="Luis", type="adult", priority="medium"),
            ],
            difficulty="medium",
            debug=args.debug,
            timeout=args.timeout,
        )

    env_scene = scene_spec_to_env_scene(scene)
    env = DisasterEnv(scene=env_scene, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        frame = env.render()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
    finally:
        env.close()

    print("Generated scene loaded into DisasterEnv.")
    print(f"Active survivor: {env_scene['active_survivor']['name']}")
    print(
        "Terrain: "
        f"seed={env_scene['terrain']['seed']}, "
        f"grid={env_scene['terrain']['grid_size']}x{env_scene['terrain']['grid_size']}"
    )
    print(f"Observation shape: {obs.shape} -> {next_obs.shape}")
    print(f"Rendered frame: {None if frame is None else frame.shape}")
    print(
        "Step result: "
        f"reward={reward:.3f}, terminated={terminated}, truncated={truncated}, "
        f"dist={info['dist']:.3f}"
    )

    json_path = scene.save_json(TEMP_DIR / "generated_scene.json")
    print(f"Wrote scene JSON: {json_path}")

    if args.save_preview:
        html_path = scene.save_preview_html(TEMP_DIR / "generated_scene.html")
        print(f"Wrote scene preview: {html_path}")


if __name__ == "__main__":
    main()
