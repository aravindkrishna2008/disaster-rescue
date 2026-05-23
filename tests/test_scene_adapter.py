from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from disaster_env import DisasterEnv, OBSTACLE_COUNT, generate_random_terrain
from scenario_agent import (
    ASSET_CATALOG,
    HazardZone,
    PlacedAsset,
    SceneSpec,
    SurvivorLocation,
    SurvivorProfile,
)
from scene_adapter import scene_spec_to_env_scene


def make_scene() -> SceneSpec:
    return SceneSpec(
        description="Generated apartment collapse",
        difficulty="medium",
        robot_start=(1.5, 1.5, 0.0),
        survivors=[
            SurvivorLocation(
                profile=SurvivorProfile(name="Luis", type="adult", priority="medium"),
                position=(18.0, 18.0, 0.0),
            ),
            SurvivorLocation(
                profile=SurvivorProfile(name="Maya", type="baby", priority="critical"),
                position=(4.0, 16.0, 0.0),
            ),
        ],
        assets=[
            PlacedAsset(
                asset_id="concrete_slab",
                position=(10.0, 10.0, ASSET_CATALOG["concrete_slab"].size[2] / 2),
                size=ASSET_CATALOG["concrete_slab"].size,
                rotation_yaw=27.0,
            ),
            PlacedAsset(
                asset_id="steel_beam",
                position=(16.0, 6.0, ASSET_CATALOG["steel_beam"].size[2] / 2),
                size=ASSET_CATALOG["steel_beam"].size,
                rotation_yaw=68.0,
            ),
            PlacedAsset(
                asset_id="rubble_pile_small",
                position=(6.0, 14.0, ASSET_CATALOG["rubble_pile_small"].size[2] / 2),
                size=ASSET_CATALOG["rubble_pile_small"].size,
                rotation_yaw=104.0,
            ),
            PlacedAsset(
                asset_id="standing_wall",
                position=(14.0, 14.0, ASSET_CATALOG["standing_wall"].size[2] / 2),
                size=ASSET_CATALOG["standing_wall"].size,
                rotation_yaw=131.0,
            ),
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


def test_scene_spec_to_env_scene_selects_priority_target_and_scales_content() -> None:
    env_scene = scene_spec_to_env_scene(make_scene())

    assert env_scene["source"] == "scenario_agent"
    assert env_scene["active_survivor"]["name"] == "Maya"
    assert env_scene["survivor_pos"] == pytest.approx([-4.8, 4.8, 0.0])
    assert env_scene["robot_start"] == pytest.approx([-6.8, -6.8, 0.0])
    assert len(env_scene["survivors"]) == 2
    assert [survivor["active"] for survivor in env_scene["survivors"]] == [False, True]

    concrete = env_scene["obstacles"][0]
    assert concrete["asset_id"] == "concrete_slab"
    assert concrete["pos"] == pytest.approx([0.0, 0.0, 0.14])
    assert concrete["size"] == pytest.approx([1.12, 0.44, 0.14])
    assert concrete["rotation_yaw"] == 27.0

    assert env_scene["hazards"][0]["type"] == "gas"
    assert env_scene["hazards"][0]["center"] == pytest.approx([-4.0, -4.0, 0.0])
    assert env_scene["hazards"][0]["radius"] == 1.0
    assert env_scene["terrain"]["grid_size"] == 10
    assert env_scene["terrain"]["height_scale"] == pytest.approx(0.5)
    assert len(env_scene["terrain"]["heights"]) == 10
    assert len(env_scene["terrain"]["roughness"]) == 10
    assert len(env_scene["terrain"]["danger"]) == 10
    assert len(env_scene["terrain"]["rigid"]) == 10


def test_disaster_env_accepts_converted_scene_with_many_assets() -> None:
    env_scene = scene_spec_to_env_scene(make_scene())
    env = DisasterEnv(scene=env_scene, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert len(env._obstacles) > OBSTACLE_COUNT

        next_obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0]))

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False
        assert "dist" in info
        assert "terrain_height" in info
        assert "terrain_roughness" in info
        assert "terrain_danger" in info
        assert "terrain_blocked" in info
    finally:
        env.close()


def test_scene_adapter_generates_deterministic_terrain_for_same_scene() -> None:
    first = scene_spec_to_env_scene(make_scene())
    second = scene_spec_to_env_scene(make_scene())

    assert first["terrain"] == second["terrain"]


def test_hard_terrain_contains_rigid_and_dangerous_cells() -> None:
    terrain = generate_random_terrain(seed=123, difficulty="hard", grid_size=10)

    assert max(max(row) for row in terrain["heights"]) == pytest.approx(0.72, abs=0.002)
    assert max(max(row) for row in terrain["roughness"]) >= 1.0
    assert sum(sum(row) for row in terrain["danger"]) > 0
    assert sum(sum(row) for row in terrain["rigid"]) > 0
