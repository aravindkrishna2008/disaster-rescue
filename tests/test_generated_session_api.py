from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from scenario_agent import ASSET_CATALOG, HazardZone, PlacedAsset, SceneSpec, SurvivorLocation, SurvivorProfile
from server import (
    GenerateSceneRequest,
    SceneRunRequest,
    _compose_scene_description,
    create_app,
)


def make_generated_scene() -> SceneSpec:
    rubble = ASSET_CATALOG["rubble_pile_small"]
    return SceneSpec(
        description="Generated collapse requiring a long approach.",
        difficulty="medium",
        robot_start=(1.0, 1.0, 0.0),
        survivors=[
            SurvivorLocation(
                profile=SurvivorProfile(name="Asha", type="child", priority="critical"),
                position=(18.0, 18.0, 0.0),
            )
        ],
        assets=[
            PlacedAsset(
                asset_id=rubble.asset_id,
                position=(16.0, 18.0, rubble.size[2] / 2),
                size=rubble.size,
            )
        ],
        hazards=[
            HazardZone(
                hazard_id="fire-1",
                type="fire",
                center=(5.0, 6.0, 0.0),
                radius=1.0,
            )
        ],
    )


def fake_episode_result(steps: int = 623) -> dict:
    return {
        "reached": True,
        "fallen": False,
        "steps": steps,
        "total_reward": 2129.59,
        "final_dist": 0.897,
        "min_dist": 0.897,
        "final_heading_error": 0.243,
        "obstacle_contacts": 0,
        "hazard_steps": 0,
        "min_obstacle_clearance": 4.202,
        "min_hazard_clearance": 2.3,
        "mean_stance_slip": 0.353,
        "mean_assist_force": 239.692,
        "gait_score": 0.683,
        "assist_scale": 0.95,
        "balance_assist_scale": 0.85,
        "trajectory": [[-7.2, -7.2, 0.0], [6.4, 6.4, 0.0]],
        "detection_event": None,
        "cancelled": False,
        "completion_reason": "target_reached",
        "frame_count": steps + 1,
        "gif_fps": 20,
        "gif_duration_seconds": (steps + 1) / 20,
        "wall_time_seconds": 3.4,
    }


def route(app, path: str, method: str):
    return next(
        item
        for item in app.routes
        if getattr(item, "path", None) == path and method in getattr(item, "methods", set())
    )


def test_generated_scene_is_retained_for_extended_console_reruns() -> None:
    app = create_app()
    generate = route(app, "/generate-scene", "POST")
    load = route(app, "/generated-scenes/{scene_id}", "GET")
    rerun = route(app, "/generated-scenes/{scene_id}/run", "POST")

    with (
        mock.patch("server.ScenarioAgent.generate_scene", return_value=make_generated_scene()),
        mock.patch("server.run_episode", return_value=fake_episode_result()) as runner,
    ):
        generated = asyncio.run(
            generate.endpoint(
                GenerateSceneRequest(
                    description="A long collapsed corridor.",
                    difficulty="medium",
                    survivor_count=1,
                )
            )
        )

        assert generated["scene_id"].startswith("generated_")
        assert generated["default_max_steps"] == 1500
        assert generated["episode"]["reached"] is True
        assert generated["episode"]["gif_duration_seconds"] == 31.2
        assert runner.call_args.kwargs["max_steps"] == 1500

        stored = load.endpoint(generated["scene_id"])
        assert stored["env_scene"]["active_survivor"]["name"] == "Asha"

        rerun_result = asyncio.run(
            rerun.endpoint(generated["scene_id"], SceneRunRequest(max_steps=2200))
        )
        assert rerun_result["episode"]["max_steps"] == 2200
        assert rerun_result["episode"]["gait_score"] == 0.683
        assert runner.call_args.kwargs["max_steps"] == 2200


def test_generate_scene_uses_theme_hint_in_world_description() -> None:
    app = create_app()
    generate = route(app, "/generate-scene", "POST")

    with (
        mock.patch("server.ScenarioAgent.generate_scene", return_value=make_generated_scene()) as generator,
        mock.patch("server.run_episode", return_value=fake_episode_result()),
    ):
        asyncio.run(
            generate.endpoint(
                GenerateSceneRequest(
                    description="One survivor trapped under a broken stairwell.",
                    difficulty="medium",
                    survivor_count=1,
                    theme="chemical plant",
                )
            )
        )

    assert generator.call_args.args[0] == (
        "chemical plant. One survivor trapped under a broken stairwell."
    )


def test_compose_scene_description_avoids_duplicate_theme() -> None:
    assert _compose_scene_description("Flooded subway tunnel with one survivor.", "flooded subway") == (
        "Flooded subway tunnel with one survivor."
    )
