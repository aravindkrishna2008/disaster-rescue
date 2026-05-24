from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents import DEFAULT_BASE_AGENT, DEFAULT_MODEL, ManagedAgent


SCENARIO_MODEL = DEFAULT_MODEL
SCENARIO_BASE_AGENT = DEFAULT_BASE_AGENT
SCENE_WIDTH_METERS = 20.0
SCENE_DEPTH_METERS = 20.0
ROBOT_RADIUS_METERS = 0.45
SURVIVOR_RADIUS_METERS = 0.35
RESERVED_PATH_WIDTH_METERS = 1.8
MAX_SURVIVOR_RUBBLE_DISTANCE_METERS = 3.0
MIN_SURVIVOR_RUBBLE_DISTANCE_METERS = 0.9
MIN_ASSETS_BY_DIFFICULTY: dict[str, int] = {
    "easy": 12,
    "medium": 18,
    "hard": 27,
}
MIN_HAZARDS_BY_DIFFICULTY: dict[str, int] = {
    "easy": 3,
    "medium": 6,
    "hard": 9,
}
VARIED_ROTATION_DEGREES = (11.0, 27.0, 43.0, 68.0, 104.0, 131.0, 157.0)

Difficulty = Literal["easy", "medium", "hard"]
Priority = Literal["low", "medium", "high", "critical"]


class AssetDefinition(BaseModel):
    asset_id: str
    label: str
    size: tuple[float, float, float]
    tags: list[str]
    color: str


ASSET_CATALOG: dict[str, AssetDefinition] = {
    "rubble_pile_small": AssetDefinition(
        asset_id="rubble_pile_small",
        label="Small rubble pile",
        size=(1.2, 1.0, 0.55),
        tags=["rubble", "obstacle"],
        color="#7b6f61",
    ),
    "rubble_pile_large": AssetDefinition(
        asset_id="rubble_pile_large",
        label="Large rubble pile",
        size=(2.2, 1.7, 0.9),
        tags=["rubble", "obstacle"],
        color="#6d6258",
    ),
    "concrete_slab": AssetDefinition(
        asset_id="concrete_slab",
        label="Concrete slab",
        size=(2.8, 1.1, 0.35),
        tags=["rubble", "obstacle"],
        color="#8e9294",
    ),
    "steel_beam": AssetDefinition(
        asset_id="steel_beam",
        label="Steel beam",
        size=(3.0, 0.35, 0.35),
        tags=["rubble", "obstacle"],
        color="#5f6870",
    ),
    "collapsed_wall": AssetDefinition(
        asset_id="collapsed_wall",
        label="Collapsed wall segment",
        size=(3.5, 0.45, 1.4),
        tags=["wall", "obstacle"],
        color="#9a9387",
    ),
    "standing_wall": AssetDefinition(
        asset_id="standing_wall",
        label="Standing wall segment",
        size=(4.0, 0.35, 2.4),
        tags=["wall", "obstacle"],
        color="#a89f93",
    ),
}


class SurvivorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    priority: Priority = "medium"


class PlacedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation_yaw: float = 0.0


class HazardZone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hazard_id: str
    type: str
    center: tuple[float, float, float]
    radius: float = Field(gt=0)
    height: float = Field(default=0.05, gt=0)
    severity: Priority = "medium"


class SurvivorLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: SurvivorProfile
    position: tuple[float, float, float]


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    difficulty: Difficulty
    bounds: tuple[float, float, float] = (SCENE_WIDTH_METERS, SCENE_DEPTH_METERS, 3.0)
    robot_start: tuple[float, float, float]
    survivors: list[SurvivorLocation]
    assets: list[PlacedAsset]
    hazards: list[HazardZone]
    notes: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path

    def save_preview_html(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(render_preview_html(self), encoding="utf-8")
        return output_path


class ScenarioAgent:
    def __init__(
        self,
        *,
        agent_id: str = "scenario-agent",
        client: Any | None = None,
        managed_agent: ManagedAgent | None = None,
    ) -> None:
        self.managed_agent = managed_agent or ManagedAgent(
            agent_id=agent_id,
            system_prompt=build_system_prompt(),
            base_agent=SCENARIO_BASE_AGENT,
            client=client,
            description="Generates validated collapsed-building rescue scenarios.",
        )

    def create(self) -> Any:
        return self.managed_agent.create()

    def generate_scene(
        self,
        description: str,
        survivor_count: int,
        survivor_profiles: list[SurvivorProfile] | None = None,
        difficulty: Difficulty = "medium",
        debug: bool = False,
        timeout: float = 45.0,
    ) -> SceneSpec:
        if survivor_count < 1:
            raise ValueError("survivor_count must be at least 1.")

        profiles = normalize_survivor_profiles(survivor_count, survivor_profiles)
        prompt = build_generation_prompt(description, survivor_count, profiles, difficulty)
        if debug:
            print("\n=== ScenarioAgent system prompt ===")
            print(self.managed_agent.system_prompt, flush=True)
            print("\n=== ScenarioAgent generation prompt ===")
            print(prompt, flush=True)

        if hasattr(self.managed_agent, "run"):
            interaction = self.managed_agent.run(prompt, timeout=timeout)
            response_text = extract_interaction_output_text(interaction)
        else:
            interaction = None
            response_text = self.managed_agent.run_text(prompt)
        if debug:
            if interaction is not None:
                print_interaction_debug(interaction)
            print("\n=== ScenarioAgent raw response ===")
            print(response_text, flush=True)

        scene = parse_scene_response(response_text)
        repaired_scene = repair_scene(scene, description, difficulty, survivor_count, profiles)
        if debug:
            print("\n=== ScenarioAgent repaired scene ===")
            print(repaired_scene.to_json(), flush=True)

        return repaired_scene


def build_system_prompt() -> str:
    catalog = [
        {
            "asset_id": asset.asset_id,
            "size": asset.size,
            "tags": asset.tags,
        }
        for asset in ASSET_CATALOG.values()
    ]
    return (
        f"You are ScenarioAgent, a managed agent configured for {SCENARIO_MODEL} "
        "style fast scene generation for rescue "
        "robot training data. Convert user disaster descriptions into 3D "
        "collapsed-building scene JSON. Use only the provided asset catalog. "
        "Do not create meshes, files, Markdown, or explanatory prose. The scene "
        "is 3D, but robot navigation is on a flat floor plane, so all walkable "
        "targets must remain reachable. Place every survivor close to rubble or "
        "collapsed debris, not in empty open space. Return only JSON matching the schema. "
        f"Asset catalog: {json.dumps(catalog)}"
    )


def build_generation_prompt(
    description: str,
    survivor_count: int,
    survivor_profiles: list[SurvivorProfile],
    difficulty: Difficulty,
) -> str:
    return json.dumps(
        {
            "task": "Generate one collapsed-building rescue training scene.",
            "description": description,
            "difficulty": difficulty,
            "bounds_meters": [SCENE_WIDTH_METERS, SCENE_DEPTH_METERS, 3.0],
            "survivor_count": survivor_count,
            "minimum_asset_count": MIN_ASSETS_BY_DIFFICULTY[difficulty],
            "minimum_hazard_count": MIN_HAZARDS_BY_DIFFICULTY[difficulty],
            "survivor_profiles": [
                profile.model_dump() for profile in survivor_profiles
            ],
            "requirements": [
                "Return only JSON.",
                "Use 3D coordinates [x, y, z] in meters.",
                "Use only asset_id values from the asset catalog.",
                (
                    f"Create at least {MIN_ASSETS_BY_DIFFICULTY[difficulty]} visible "
                    "solid assets. This is intentionally dense rubble."
                ),
                (
                    f"Create at least {MIN_HAZARDS_BY_DIFFICULTY[difficulty]} hazard "
                    "zones for this difficulty."
                ),
                (
                    "Use varied non-cardinal rotation_yaw values. Do not align most "
                    "assets to only 0, 90, or 180 degrees."
                ),
                "Preserve survivor profile name, type, and priority exactly.",
                "Place all solid assets on the floor with z equal to half height.",
                "Place all survivors on the floor with z equal to 0.",
                (
                    "Place every survivor within "
                    f"{MAX_SURVIVOR_RUBBLE_DISTANCE_METERS:.1f} meters of at least one "
                    "rubble/debris asset, but not overlapping the asset footprint."
                ),
                "Include robot_start, survivors, assets, hazards, and notes.",
                "Make every survivor reachable from robot_start.",
            ],
            "schema": {
                "description": "string",
                "difficulty": "easy|medium|hard",
                "bounds": [20.0, 20.0, 3.0],
                "robot_start": [1.5, 1.5, 0.0],
                "survivors": [
                    {
                        "profile": {
                            "name": "string",
                            "type": "string",
                            "priority": "low|medium|high|critical",
                        },
                        "position": [18.0, 18.0, 0.0],
                    }
                ],
                "assets": [
                    {
                        "asset_id": "rubble_pile_small",
                        "position": [5.0, 6.0, 0.275],
                        "size": [1.2, 1.0, 0.55],
                        "rotation_yaw": 0.0,
                    }
                ],
                "hazards": [
                    {
                        "hazard_id": "hazard-1",
                        "type": "fire|gas|unstable_floor",
                        "center": [8.0, 9.0, 0.0],
                        "radius": 1.0,
                        "height": 0.05,
                        "severity": "medium",
                    }
                ],
                "notes": "string",
            },
        }
    )


def normalize_survivor_profiles(
    survivor_count: int,
    survivor_profiles: list[SurvivorProfile] | None,
) -> list[SurvivorProfile]:
    profiles = list(survivor_profiles or [])
    normalized = profiles[:survivor_count]
    for index in range(len(normalized), survivor_count):
        normalized.append(
            SurvivorProfile(
                name=f"Survivor {index + 1}",
                type="adult",
                priority="medium",
            )
        )
    return normalized


def parse_scene_response(response_text: str) -> SceneSpec:
    payload = extract_json_object(response_text)
    return SceneSpec.model_validate(normalize_scene_payload(json.loads(payload)))


def normalize_scene_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill omitted catalog-backed fields before strict SceneSpec validation."""
    normalized = dict(payload)
    assets = []
    for asset in normalized.get("assets", []) or []:
        if not isinstance(asset, dict):
            assets.append(asset)
            continue
        normalized_asset = dict(asset)
        asset_id = str(normalized_asset.get("asset_id", "rubble_pile_small"))
        definition = ASSET_CATALOG.get(asset_id, ASSET_CATALOG["rubble_pile_small"])
        normalized_asset.setdefault("asset_id", definition.asset_id)
        normalized_asset.setdefault("size", definition.size)
        assets.append(normalized_asset)
    normalized["assets"] = assets

    hazards = []
    for index, hazard in enumerate(normalized.get("hazards", []) or []):
        if not isinstance(hazard, dict):
            hazards.append(hazard)
            continue
        normalized_hazard = dict(hazard)
        normalized_hazard.setdefault("hazard_id", f"hazard-{index + 1}")
        normalized_hazard.setdefault("type", "unstable_floor")
        normalized_hazard.setdefault("center", (8.0, 8.0, 0.0))
        normalized_hazard.setdefault("radius", 1.0)
        normalized_hazard.setdefault("severity", "medium")
        try:
            height = float(normalized_hazard.get("height", 0.05))
        except (TypeError, ValueError):
            height = 0.05
        normalized_hazard["height"] = height if height > 0 else 0.05
        hazards.append(normalized_hazard)
    normalized["hazards"] = hazards
    return normalized


def extract_interaction_output_text(interaction: Any) -> str:
    output_text = getattr(interaction, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                chunks.append(str(text))

    if chunks:
        return "\n".join(chunks)

    status = getattr(interaction, "status", "unknown")
    raise RuntimeError(f"Managed agent interaction did not return text. Status: {status}")


def print_interaction_debug(interaction: Any) -> None:
    print("\n=== ScenarioAgent interaction status ===", flush=True)
    print(f"id: {getattr(interaction, 'id', None)}", flush=True)
    print(f"status: {getattr(interaction, 'status', None)}", flush=True)

    thought_chunks: list[str] = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) != "thought":
            continue
        for item in getattr(step, "summary", []) or []:
            text = getattr(item, "text", None)
            if text:
                thought_chunks.append(str(text).strip())

    if thought_chunks:
        print("\n=== ScenarioAgent thought summaries ===", flush=True)
        print("\n\n".join(thought_chunks), flush=True)


def extract_json_object(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("ScenarioAgent response did not contain a JSON object.")
    return stripped[start : end + 1]


def repair_scene(
    scene: SceneSpec,
    description: str,
    difficulty: Difficulty,
    survivor_count: int,
    survivor_profiles: list[SurvivorProfile],
) -> SceneSpec:
    robot_start = clamp_point(scene.robot_start, radius=ROBOT_RADIUS_METERS)
    robot_start = (robot_start[0], robot_start[1], 0.0)

    normalized_assets = normalize_assets(scene.assets)
    survivors = repair_survivors(
        scene.survivors,
        survivor_count,
        survivor_profiles,
        robot_start,
        normalized_assets,
    )
    path_corridors = build_reserved_corridors(
        robot_start,
        [survivor.position for survivor in survivors],
    )

    assets = repair_assets(normalized_assets, path_corridors, robot_start, survivors)
    hazards = repair_hazards(scene.hazards, path_corridors)
    assets = ensure_asset_density(assets, difficulty, path_corridors, robot_start, survivors)
    hazards = ensure_hazard_density(hazards, difficulty, path_corridors)
    survivors = repair_survivors_near_rubble(survivors, assets, robot_start)

    return SceneSpec(
        description=description,
        difficulty=difficulty,
        bounds=(SCENE_WIDTH_METERS, SCENE_DEPTH_METERS, 3.0),
        robot_start=robot_start,
        survivors=survivors,
        assets=assets,
        hazards=hazards,
        notes=scene.notes,
    )


def repair_survivors(
    survivors: list[SurvivorLocation],
    survivor_count: int,
    profiles: list[SurvivorProfile],
    robot_start: tuple[float, float, float],
    assets: list[PlacedAsset],
) -> list[SurvivorLocation]:
    preferred_positions = [
        (18.0, 18.0, 0.0),
        (18.0, 2.0, 0.0),
        (2.0, 18.0, 0.0),
        (15.0, 15.0, 0.0),
        (15.0, 5.0, 0.0),
        (5.0, 15.0, 0.0),
    ]
    repaired: list[SurvivorLocation] = []

    for index, profile in enumerate(profiles[:survivor_count]):
        proposed = (
            survivors[index].position
            if index < len(survivors)
            else preferred_positions[index % len(preferred_positions)]
        )
        position = clamp_point(proposed, radius=SURVIVOR_RADIUS_METERS)
        position = (position[0], position[1], 0.0)

        if distance_2d(position, robot_start) < 2.0:
            position = preferred_positions[index % len(preferred_positions)]
        position = position_near_rubble(position, assets, index)

        while any(distance_2d(position, other.position) < 1.2 for other in repaired):
            position = next_open_point(position, repaired)

        repaired.append(SurvivorLocation(profile=profile, position=position))

    return repaired


def normalize_assets(assets: list[PlacedAsset]) -> list[PlacedAsset]:
    normalized: list[PlacedAsset] = []
    for index, asset in enumerate(assets):
        definition = ASSET_CATALOG.get(asset.asset_id, ASSET_CATALOG["rubble_pile_small"])
        normalized.append(
            PlacedAsset(
                asset_id=definition.asset_id,
                position=clamp_asset_position(asset.position, definition.size),
                size=definition.size,
                rotation_yaw=normalize_rotation(asset.rotation_yaw, index),
            )
        )
    return normalized


def repair_assets(
    assets: list[PlacedAsset],
    path_corridors: list[tuple[float, float, float, float]],
    robot_start: tuple[float, float, float],
    survivors: list[SurvivorLocation],
) -> list[PlacedAsset]:
    repaired: list[PlacedAsset] = []
    protected_points = [robot_start, *[survivor.position for survivor in survivors]]

    for asset in assets:
        candidate = asset

        if asset_conflicts(candidate, repaired, path_corridors, protected_points):
            replacement = find_asset_position(candidate, repaired, path_corridors, protected_points)
            if replacement is None:
                continue
            candidate = replacement

        repaired.append(candidate)

    return repaired


def ensure_asset_density(
    assets: list[PlacedAsset],
    difficulty: Difficulty,
    path_corridors: list[tuple[float, float, float, float]],
    robot_start: tuple[float, float, float],
    survivors: list[SurvivorLocation],
) -> list[PlacedAsset]:
    target_count = MIN_ASSETS_BY_DIFFICULTY[difficulty]
    repaired = list(assets)
    protected_points = [robot_start, *[survivor.position for survivor in survivors]]
    asset_cycle = [
        "rubble_pile_small",
        "rubble_pile_large",
        "concrete_slab",
        "steel_beam",
        "collapsed_wall",
        "standing_wall",
    ]
    cursor = 0

    for y in range(2, 19, 2):
        for x in range(2, 19, 2):
            if len(repaired) >= target_count:
                return repaired
            asset_id = asset_cycle[cursor % len(asset_cycle)]
            definition = ASSET_CATALOG[asset_id]
            candidate = PlacedAsset(
                asset_id=asset_id,
                position=clamp_asset_position((float(x), float(y), 0.0), definition.size),
                size=definition.size,
                rotation_yaw=VARIED_ROTATION_DEGREES[cursor % len(VARIED_ROTATION_DEGREES)],
            )
            cursor += 1
            if not asset_conflicts(candidate, repaired, path_corridors, protected_points):
                repaired.append(candidate)

    return repaired


def repair_survivors_near_rubble(
    survivors: list[SurvivorLocation],
    assets: list[PlacedAsset],
    robot_start: tuple[float, float, float],
) -> list[SurvivorLocation]:
    repaired: list[SurvivorLocation] = []
    for index, survivor in enumerate(survivors):
        position = position_near_rubble(survivor.position, assets, index)
        if distance_2d(position, robot_start) < 2.0:
            position = position_near_rubble((18.0, 18.0, 0.0), assets, index)
        while any(distance_2d(position, other.position) < 1.2 for other in repaired):
            position = next_open_point(position, repaired)
            position = position_near_rubble(position, assets, index)
        repaired.append(SurvivorLocation(profile=survivor.profile, position=position))
    return repaired


def position_near_rubble(
    position: tuple[float, float, float],
    assets: list[PlacedAsset],
    index: int,
) -> tuple[float, float, float]:
    rubble_assets = [asset for asset in assets if is_rubble_asset(asset)]
    if not rubble_assets:
        return (position[0], position[1], 0.0)

    nearest = min(rubble_assets, key=lambda asset: distance_2d(position, asset.position))
    if distance_2d(position, nearest.position) <= MAX_SURVIVOR_RUBBLE_DISTANCE_METERS:
        return (position[0], position[1], 0.0)

    angle = (index % 8) * (math.pi / 4)
    clearance = max(
        MIN_SURVIVOR_RUBBLE_DISTANCE_METERS,
        max(nearest.size[0], nearest.size[1]) / 2 + SURVIVOR_RADIUS_METERS + 0.35,
    )
    candidate = (
        nearest.position[0] + math.cos(angle) * clearance,
        nearest.position[1] + math.sin(angle) * clearance,
        0.0,
    )
    clamped = clamp_point(candidate, radius=SURVIVOR_RADIUS_METERS)
    return (clamped[0], clamped[1], 0.0)


def is_rubble_asset(asset: PlacedAsset) -> bool:
    definition = ASSET_CATALOG.get(asset.asset_id)
    if definition is None:
        return False
    return "rubble" in definition.tags


def nearest_rubble_distance(
    position: tuple[float, float, float],
    assets: list[PlacedAsset],
) -> float | None:
    rubble_assets = [asset for asset in assets if is_rubble_asset(asset)]
    if not rubble_assets:
        return None
    return min(distance_2d(position, asset.position) for asset in rubble_assets)


def normalize_rotation(rotation_yaw: float, index: int) -> float:
    normalized = rotation_yaw % 180
    if is_cardinal_rotation(normalized):
        return VARIED_ROTATION_DEGREES[index % len(VARIED_ROTATION_DEGREES)]
    return normalized


def is_cardinal_rotation(rotation_yaw: float) -> bool:
    return any(abs(rotation_yaw - angle) <= 5 for angle in (0, 90, 180))


def ensure_hazard_density(
    hazards: list[HazardZone],
    difficulty: Difficulty,
    path_corridors: list[tuple[float, float, float, float]],
) -> list[HazardZone]:
    target_count = MIN_HAZARDS_BY_DIFFICULTY[difficulty]
    repaired = list(hazards)
    hazard_types = ["fire", "gas", "unstable_floor"]
    severities: list[Priority] = ["medium", "high", "critical"]
    cursor = 0

    for y in range(3, 18, 3):
        for x in range(3, 18, 3):
            if len(repaired) >= target_count:
                return repaired
            radius = 0.7 + (cursor % 3) * 0.25
            center = (float(x), float(y), 0.0)
            if circle_conflicts_corridors(center, radius, path_corridors):
                cursor += 1
                continue
            if any(
                distance_2d(center, hazard.center) < hazard.radius + radius + 0.8
                for hazard in repaired
            ):
                cursor += 1
                continue

            repaired.append(
                HazardZone(
                    hazard_id=f"generated-hazard-{len(repaired) + 1}",
                    type=hazard_types[cursor % len(hazard_types)],
                    center=center,
                    radius=radius,
                    height=0.05,
                    severity=severities[cursor % len(severities)],
                )
            )
            cursor += 1

    return repaired


def repair_hazards(
    hazards: list[HazardZone],
    path_corridors: list[tuple[float, float, float, float]],
) -> list[HazardZone]:
    repaired: list[HazardZone] = []
    for index, hazard in enumerate(hazards):
        radius = max(0.4, min(hazard.radius, 2.2))
        center = clamp_point(hazard.center, radius=radius)
        candidate = HazardZone(
            hazard_id=hazard.hazard_id or f"hazard-{index + 1}",
            type=hazard.type,
            center=(center[0], center[1], 0.0),
            radius=radius,
            height=hazard.height,
            severity=hazard.severity,
        )
        if circle_conflicts_corridors(candidate.center, candidate.radius, path_corridors):
            moved = find_hazard_position(candidate, path_corridors)
            if moved is None:
                continue
            candidate = moved
        repaired.append(candidate)
    return repaired


def clamp_point(
    point: tuple[float, float, float],
    *,
    radius: float,
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        min(max(float(x), radius), SCENE_WIDTH_METERS - radius),
        min(max(float(y), radius), SCENE_DEPTH_METERS - radius),
        float(z),
    )


def clamp_asset_position(
    point: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    width, depth, height = size
    x, y, _ = point
    return (
        min(max(float(x), width / 2), SCENE_WIDTH_METERS - width / 2),
        min(max(float(y), depth / 2), SCENE_DEPTH_METERS - depth / 2),
        height / 2,
    )


def build_reserved_corridors(
    robot_start: tuple[float, float, float],
    survivor_positions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    corridors = []
    half_width = RESERVED_PATH_WIDTH_METERS / 2
    rx, ry, _ = robot_start
    for sx, sy, _ in survivor_positions:
        corridors.append(
            (
                min(rx, sx) - half_width,
                ry - half_width,
                max(rx, sx) + half_width,
                ry + half_width,
            )
        )
        corridors.append(
            (
                sx - half_width,
                min(ry, sy) - half_width,
                sx + half_width,
                max(ry, sy) + half_width,
            )
        )
    return corridors


def asset_conflicts(
    asset: PlacedAsset,
    repaired_assets: list[PlacedAsset],
    path_corridors: list[tuple[float, float, float, float]],
    protected_points: list[tuple[float, float, float]],
) -> bool:
    rect = asset_rect(asset)
    if any(rects_intersect(rect, corridor) for corridor in path_corridors):
        return True
    if any(rects_intersect(rect, asset_rect(other)) for other in repaired_assets):
        return True
    return any(point_inside_rect(point, expand_rect(rect, 0.6)) for point in protected_points)


def find_asset_position(
    asset: PlacedAsset,
    repaired_assets: list[PlacedAsset],
    path_corridors: list[tuple[float, float, float, float]],
    protected_points: list[tuple[float, float, float]],
) -> PlacedAsset | None:
    for x in range(2, 19, 2):
        for y in range(2, 19, 2):
            candidate = asset.model_copy(
                update={"position": clamp_asset_position((x, y, 0.0), asset.size)}
            )
            if not asset_conflicts(candidate, repaired_assets, path_corridors, protected_points):
                return candidate
    return None


def find_hazard_position(
    hazard: HazardZone,
    path_corridors: list[tuple[float, float, float, float]],
) -> HazardZone | None:
    for x in range(3, 18, 3):
        for y in range(3, 18, 3):
            if not circle_conflicts_corridors((float(x), float(y), 0.0), hazard.radius, path_corridors):
                return hazard.model_copy(update={"center": (float(x), float(y), 0.0)})
    return None


def asset_rect(asset: PlacedAsset) -> tuple[float, float, float, float]:
    x, y, _ = asset.position
    width, depth, _ = asset.size
    return (x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)


def rects_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def expand_rect(
    rect: tuple[float, float, float, float],
    amount: float,
) -> tuple[float, float, float, float]:
    return (
        rect[0] - amount,
        rect[1] - amount,
        rect[2] + amount,
        rect[3] + amount,
    )


def point_inside_rect(
    point: tuple[float, float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    x, y, _ = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def circle_conflicts_corridors(
    center: tuple[float, float, float],
    radius: float,
    corridors: list[tuple[float, float, float, float]],
) -> bool:
    x, y, _ = center
    for rect in corridors:
        closest_x = min(max(x, rect[0]), rect[2])
        closest_y = min(max(y, rect[1]), rect[3])
        if ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5 <= radius:
            return True
    return False


def distance_2d(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def next_open_point(
    point: tuple[float, float, float],
    occupied: list[SurvivorLocation],
) -> tuple[float, float, float]:
    x, y, _ = point
    for offset in range(1, 10):
        candidate = clamp_point((x - offset, y, 0.0), radius=SURVIVOR_RADIUS_METERS)
        if all(distance_2d(candidate, other.position) >= 1.2 for other in occupied):
            return (candidate[0], candidate[1], 0.0)
    return (x, y, 0.0)


def render_preview_html(scene: SceneSpec) -> str:
    scene_svg = render_scene_svg(scene)
    survivor_list = "\n".join(
        "<li>"
        f"<strong>{html.escape(survivor.profile.name)}</strong> "
        f"({html.escape(survivor.profile.type)}, "
        f"{html.escape(survivor.profile.priority)}) "
        f"at {format_point(survivor.position)}"
        "</li>"
        for survivor in scene.survivors
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ScenarioAgent Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f1ea;
      color: #1f2933;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: 100vh;
    }}
    .stage-wrap {{
      padding: 24px;
      overflow: auto;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    svg {{
      width: min(100%, 1180px);
      height: auto;
      filter: drop-shadow(0 18px 36px rgba(31, 41, 51, 0.16));
    }}
    .floor {{ fill: #ded7cb; stroke: #8f8678; stroke-width: 1.6; }}
    .grid {{ stroke: rgba(31, 41, 51, 0.11); stroke-width: 1; }}
    .asset-face {{ stroke: rgba(31, 41, 51, 0.5); stroke-width: 1.4; }}
    .asset-top {{ stroke: rgba(31, 41, 51, 0.62); stroke-width: 1.6; }}
    .height-label {{
      font-size: 12px;
      font-weight: 700;
      fill: #1f2933;
      paint-order: stroke;
      stroke: white;
      stroke-width: 4px;
    }}
    .point-label {{
      font-size: 13px;
      font-weight: 700;
      fill: #1f2933;
      paint-order: stroke;
      stroke: white;
      stroke-width: 4px;
    }}
    aside {{
      padding: 24px;
      background: #ffffff;
      border-left: 1px solid #d5d0c7;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    h2 {{
      margin: 24px 0 8px;
      font-size: 15px;
    }}
    p, li {{
      font-size: 14px;
      line-height: 1.45;
    }}
    code {{
      background: #eef0f2;
      padding: 2px 5px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main class="stage-wrap">
      {scene_svg}
    </main>
    <aside>
      <h1>ScenarioAgent Preview</h1>
      <p><strong>Difficulty:</strong> {html.escape(scene.difficulty)}</p>
      <p><strong>Bounds:</strong> {format_point(scene.bounds)} meters</p>
      <p><strong>Robot:</strong> {format_point(scene.robot_start)}</p>
      <h2>Description</h2>
      <p>{html.escape(scene.description)}</p>
      <h2>Survivors</h2>
      <ul>{survivor_list}</ul>
      <h2>Scene Counts</h2>
      <p><code>{len(scene.assets)}</code> assets, <code>{len(scene.hazards)}</code> hazards</p>
      <h2>Notes</h2>
      <p>{html.escape(scene.notes or "No notes provided.")}</p>
    </aside>
  </div>
</body>
</html>
"""


def render_scene_svg(scene: SceneSpec) -> str:
    floor = svg_polygon(
        [project_iso(0, 0, 0), project_iso(SCENE_WIDTH_METERS, 0, 0),
         project_iso(SCENE_WIDTH_METERS, SCENE_DEPTH_METERS, 0),
         project_iso(0, SCENE_DEPTH_METERS, 0)],
        'class="floor"',
    )
    grid_lines = "\n".join(render_grid_line(index) for index in range(1, 20))
    hazards = "\n".join(render_hazard(hazard) for hazard in scene.hazards)
    assets = "\n".join(
        render_asset(asset)
        for asset in sorted(scene.assets, key=lambda item: item.position[0] + item.position[1])
    )
    survivors = "\n".join(render_survivor(survivor) for survivor in scene.survivors)
    robot = render_robot(scene.robot_start)
    return f"""<svg viewBox="0 0 1280 820" role="img" aria-label="Generated rescue scenario preview">
  <rect x="0" y="0" width="1280" height="820" fill="#f4f1ea"/>
  {floor}
  {grid_lines}
  {hazards}
  {assets}
  {survivors}
  {robot}
</svg>"""


def render_grid_line(index: int) -> str:
    horizontal = svg_polyline(
        [project_iso(0, float(index), 0), project_iso(SCENE_WIDTH_METERS, float(index), 0)],
        'class="grid"',
    )
    vertical = svg_polyline(
        [project_iso(float(index), 0, 0), project_iso(float(index), SCENE_DEPTH_METERS, 0)],
        'class="grid"',
    )
    return f"{horizontal}\n{vertical}"


def render_asset(asset: PlacedAsset) -> str:
    x, y, _ = asset.position
    width, depth, height = asset.size
    definition = ASSET_CATALOG.get(asset.asset_id, ASSET_CATALOG["rubble_pile_small"])
    title = f"{definition.label} ({height:.1f}m tall)"
    bottom = rotated_box_corners(x, y, width, depth, asset.rotation_yaw)
    top = [(px, py, height) for px, py, _ in bottom]

    top_points = [project_iso(*point) for point in top]
    front_points = [project_iso(*top[1]), project_iso(*top[2]), project_iso(*bottom[2]), project_iso(*bottom[1])]
    side_points = [project_iso(*top[2]), project_iso(*top[3]), project_iso(*bottom[3]), project_iso(*bottom[2])]
    label_x, label_y = project_iso(x, y, height + 0.25)
    return "\n".join(
        [
            f"<g><title>{html.escape(title)}</title>",
            svg_polygon(side_points, f'class="asset-face" fill="{shade_color(definition.color, -18)}"'),
            svg_polygon(front_points, f'class="asset-face" fill="{shade_color(definition.color, -28)}"'),
            svg_polygon(top_points, f'class="asset-top" fill="{definition.color}"'),
            f'<text class="height-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle">{height:.1f}m</text>',
            "</g>",
        ]
    )


def render_hazard(hazard: HazardZone) -> str:
    x, y, _ = hazard.center
    cx, cy = project_iso(x, y, 0)
    rx = hazard.radius * 28
    ry = hazard.radius * 16
    return (
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
        'fill="#e04f39" fill-opacity="0.24" stroke="#9f2f22" '
        f'stroke-width="2" stroke-dasharray="7 5"><title>{html.escape(hazard.type)}</title></ellipse>'
    )


def render_survivor(survivor: SurvivorLocation) -> str:
    x, y, _ = survivor.position
    cx, cy = project_iso(x, y, 0.15)
    label = html.escape(survivor.profile.name)
    return (
        f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="#f4c430" '
        f'stroke="#7a5c00" stroke-width="3"><title>{label}</title></circle>'
        f'<text class="point-label" x="{cx + 13:.1f}" y="{cy - 8:.1f}">{label}</text></g>'
    )


def render_robot(position: tuple[float, float, float]) -> str:
    x, y, _ = position
    cx, cy = project_iso(x, y, 0.15)
    return (
        f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="#2563eb" '
        'stroke="#1e3a8a" stroke-width="3"><title>Robot start</title></circle>'
        f'<text class="point-label" x="{cx + 14:.1f}" y="{cy - 9:.1f}">Robot</text></g>'
    )


def rotated_box_corners(
    x: float,
    y: float,
    width: float,
    depth: float,
    yaw_degrees: float,
) -> list[tuple[float, float, float]]:
    radians = yaw_degrees * 3.141592653589793 / 180
    cos_yaw = math.cos(radians)
    sin_yaw = math.sin(radians)
    corners = [
        (-width / 2, -depth / 2),
        (width / 2, -depth / 2),
        (width / 2, depth / 2),
        (-width / 2, depth / 2),
    ]
    return [
        (
            x + local_x * cos_yaw - local_y * sin_yaw,
            y + local_x * sin_yaw + local_y * cos_yaw,
            0.0,
        )
        for local_x, local_y in corners
    ]


def project_iso(x: float, y: float, z: float) -> tuple[float, float]:
    return (640 + (x - y) * 28, 90 + (x + y) * 16 - z * 34)


def svg_polygon(points: list[tuple[float, float]], attributes: str) -> str:
    return f'<polygon points="{svg_points(points)}" {attributes}/>'


def svg_polyline(points: list[tuple[float, float]], attributes: str) -> str:
    return f'<polyline points="{svg_points(points)}" fill="none" {attributes}/>'


def svg_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def shade_color(hex_color: str, amount: int) -> str:
    color = hex_color.lstrip("#")
    red = min(255, max(0, int(color[0:2], 16) + amount))
    green = min(255, max(0, int(color[2:4], 16) + amount))
    blue = min(255, max(0, int(color[4:6], 16) + amount))
    return f"#{red:02x}{green:02x}{blue:02x}"


def format_point(point: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.1f}" for value in point) + "]"
