from __future__ import annotations

from math import radians
from typing import Any

from scenario_agent import ASSET_CATALOG, SceneSpec, SurvivorLocation


ENV_SCALE = 0.8
SCENE_CENTER = 10.0
PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def scene_spec_to_env_scene(scene: SceneSpec) -> dict[str, Any]:
    """Convert ScenarioAgent output into the dict shape consumed by DisasterEnv."""
    active_survivor = _select_active_survivor(scene.survivors)
    active_position = _to_env_floor_point(active_survivor.position)

    survivors = [
        {
            "name": survivor.profile.name,
            "type": survivor.profile.type,
            "priority": survivor.profile.priority,
            "pos": _to_env_floor_point(survivor.position),
            "active": survivor is active_survivor,
        }
        for survivor in scene.survivors
    ]

    obstacles = []
    for asset in scene.assets:
        definition = ASSET_CATALOG.get(asset.asset_id, ASSET_CATALOG["rubble_pile_small"])
        width, depth, height = definition.size
        half_size = [
            width * ENV_SCALE / 2.0,
            depth * ENV_SCALE / 2.0,
            height * ENV_SCALE / 2.0,
        ]
        x, y = _to_env_xy(asset.position)
        obstacles.append(
            {
                "asset_id": definition.asset_id,
                "pos": [x, y, half_size[2]],
                "size": half_size,
                "rotation_yaw": float(asset.rotation_yaw),
                "rotation_yaw_rad": radians(float(asset.rotation_yaw)),
            }
        )

    hazards = [
        {
            "hazard_id": hazard.hazard_id,
            "type": hazard.type,
            "center": _to_env_floor_point(hazard.center),
            "radius": float(hazard.radius) * ENV_SCALE,
            "height": float(hazard.height) * ENV_SCALE,
            "severity": hazard.severity,
        }
        for hazard in scene.hazards
    ]

    return {
        "description": scene.description,
        "difficulty": scene.difficulty,
        "robot_start": _to_env_floor_point(scene.robot_start),
        "survivor_pos": active_position,
        "active_survivor": {
            "name": active_survivor.profile.name,
            "type": active_survivor.profile.type,
            "priority": active_survivor.profile.priority,
            "pos": active_position,
        },
        "survivors": survivors,
        "obstacles": obstacles,
        "hazards": hazards,
        "notes": scene.notes,
        "source": "scenario_agent",
    }


def _select_active_survivor(survivors: list[SurvivorLocation]) -> SurvivorLocation:
    if not survivors:
        raise ValueError("SceneSpec must include at least one survivor.")
    return min(
        enumerate(survivors),
        key=lambda item: (PRIORITY_RANK.get(item[1].profile.priority, 99), item[0]),
    )[1]


def _to_env_xy(point: tuple[float, float, float]) -> tuple[float, float]:
    x, y, _ = point
    return ((float(x) - SCENE_CENTER) * ENV_SCALE, (float(y) - SCENE_CENTER) * ENV_SCALE)


def _to_env_floor_point(point: tuple[float, float, float]) -> list[float]:
    x, y = _to_env_xy(point)
    return [x, y, 0.0]
