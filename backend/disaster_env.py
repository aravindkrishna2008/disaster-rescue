"""
Grounded disaster rescue environment.

The rendered robot is humanoid-shaped, but the policy controls a grounded 2D
body. It cannot fly: there is no z action, no z joint, and the reward/distance
are computed only on the floor plane. Obstacles are visible MuJoCo boxes and
are enforced by Python collision checks so the policy must route around them.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

DEFAULT_SCENE = {
    "robot_start": [-4.0, -4.0, 0.0],
    "survivor_pos": [4.0, 4.0, 0.0],
    "terrain": {
        "seed": 7,
        "grid_size": 10,
        "height_scale": 0.5,
        "roughness_scale": 0.9,
    },
    "obstacles": [
        {"pos": [0.0, 0.0, 1.0], "size": [0.5, 0.5, 1.0]},
        {"pos": [2.0, -2.0, 0.75], "size": [0.4, 0.4, 0.75]},
        {"pos": [-2.0, 2.0, 1.25], "size": [0.4, 0.4, 1.25]},
    ],
    "hazards": [
        {"center": [1.0, 1.0, 0.0], "radius": 1.2},
        {"center": [-1.5, -1.0, 0.0], "radius": 0.9},
    ],
    "difficulty": "medium",
}

REACH_DIST = 0.75
REACH_BONUS = 200.0
HAZARD_PEN = 1.0
COLLISION_PEN = 8.0
DETECTION_BONUS = 5.0
MAX_STEPS = 600
WORLD_HALF = 8.0
ACTION_SPEED = 0.18
ROBOT_RADIUS = 0.35
GROUND_Z = 0.0
OBSTACLE_COUNT = 3
G1_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
HUMANOID_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mujoco_menagerie" / "humanoid" / "humanoid.xml"

ASSET_RGBA = {
    "rubble_pile_small": "0.48 0.43 0.38 1",
    "rubble_pile_large": "0.42 0.38 0.34 1",
    "concrete_slab": "0.56 0.57 0.58 1",
    "steel_beam": "0.38 0.41 0.44 1",
    "collapsed_wall": "0.60 0.57 0.53 1",
    "standing_wall": "0.66 0.62 0.58 1",
}

HAZARD_RGBA = {
    "fire": "1.0 0.18 0.05 0.38",
    "gas": "0.25 0.85 0.45 0.30",
    "unstable_floor": "1.0 0.75 0.15 0.34",
}


def generate_random_terrain(
    *,
    seed: int | None = None,
    difficulty: str = "medium",
    grid_size: int = 10,
) -> dict:
    """Create lightweight terrain metadata for the Gymnasium environment."""
    difficulty_scales = {
        "easy": (0.28, 0.55, 0.08),
        "medium": (0.5, 0.9, 0.14),
        "hard": (0.72, 1.2, 0.22),
    }
    height_scale, roughness_scale, void_rate = difficulty_scales.get(
        difficulty,
        difficulty_scales["medium"],
    )
    rng = np.random.default_rng(seed)
    heights = rng.uniform(0.0, 1.0, size=(grid_size, grid_size))

    for _ in range(2):
        padded = np.pad(heights, 1, mode="edge")
        heights = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1] * 2
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 10.0

    ridge_count = max(2, grid_size // 4)
    for _ in range(ridge_count):
        row = int(rng.integers(0, grid_size))
        col = int(rng.integers(0, grid_size))
        if rng.random() < 0.5:
            heights[row, :] += rng.uniform(0.45, 0.9)
            heights[max(0, row - 1) : min(grid_size, row + 2), :] += rng.uniform(0.1, 0.25)
        else:
            heights[:, col] += rng.uniform(0.45, 0.9)
            heights[:, max(0, col - 1) : min(grid_size, col + 2)] += rng.uniform(0.1, 0.25)

    slab_count = max(3, grid_size // 2)
    rigid = np.zeros_like(heights)
    for _ in range(slab_count):
        width = int(rng.integers(1, 4))
        depth = int(rng.integers(1, 3))
        row = int(rng.integers(0, max(1, grid_size - depth)))
        col = int(rng.integers(0, max(1, grid_size - width)))
        slab_height = rng.uniform(0.45, 1.0)
        heights[row : row + depth, col : col + width] += slab_height
        rigid[row : row + depth, col : col + width] = 1.0

    heights -= float(np.min(heights))
    peak = float(np.max(heights))
    if peak > 0:
        heights = heights / peak

    roughness = np.zeros_like(heights)
    roughness[1:, :] = np.maximum(roughness[1:, :], np.abs(heights[1:, :] - heights[:-1, :]))
    roughness[:, 1:] = np.maximum(roughness[:, 1:], np.abs(heights[:, 1:] - heights[:, :-1]))
    rough_peak = float(np.max(roughness))
    if rough_peak > 0:
        roughness = roughness / rough_peak

    danger = (rng.random((grid_size, grid_size)) < void_rate).astype(float)
    danger = np.maximum(danger, rigid * 0.35)
    roughness = np.maximum(roughness, danger * 0.9)

    return {
        "seed": seed,
        "grid_size": grid_size,
        "height_scale": height_scale,
        "roughness_scale": roughness_scale,
        "void_rate": void_rate,
        "heights": np.round(heights * height_scale, 3).tolist(),
        "roughness": np.round(roughness * roughness_scale, 3).tolist(),
        "danger": np.round(danger, 3).tolist(),
        "rigid": np.round(rigid, 3).tolist(),
    }


def _vec3(seq, default_z=0.0) -> list[float]:
    seq = list(seq)
    if len(seq) == 2:
        seq.append(default_z)
    return [float(v) for v in seq[:3]]


def _xy(seq) -> np.ndarray:
    return np.array(_vec3(seq)[:2], dtype=np.float64)


def _obstacle_bounds(obstacle: dict) -> tuple[np.ndarray, np.ndarray]:
    center = _xy(obstacle["pos"])
    size = list(obstacle.get("size", [0.5, 0.5, 1.0]))
    if len(size) == 1:
        size = [size[0], size[0], 1.0]
    half = np.array(size[:2], dtype=np.float64)
    return center, half


def _xml_name(prefix: str, index: int, raw: object | None = None) -> str:
    suffix = "" if raw is None else "_" + "".join(
        char if char.isalnum() or char == "_" else "_" for char in str(raw)
    )
    return f"{prefix}_{index}{suffix}"


def _obstacle_sort_key(pos: np.ndarray, obstacle: tuple[np.ndarray, np.ndarray]) -> float:
    center, half = obstacle
    outside = np.maximum(np.abs(pos - center) - (half + ROBOT_RADIUS), 0.0)
    return float(np.linalg.norm(outside))


def _remove_xml_children(parent: ET.Element, tags: set[str]) -> None:
    for child in list(parent):
        if child.tag in tags:
            parent.remove(child)
        else:
            _remove_xml_children(child, tags)


def _prepare_g1_robot_xml() -> tuple[str, str, str]:
    """Load Unitree G1 as a fixed visual robot mounted on the policy body."""
    root = ET.parse(G1_MODEL_PATH).getroot()
    default = root.find("default")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if default is None or asset is None or worldbody is None:
        raise ValueError(f"Invalid Unitree G1 MJCF: {G1_MODEL_PATH}")

    g1_body = worldbody.find("body[@name='pelvis']")
    if g1_body is None:
        raise ValueError(f"Unitree G1 pelvis body not found: {G1_MODEL_PATH}")

    asset_dir = G1_MODEL_PATH.parent / "assets"
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh.set("file", str(asset_dir / mesh_file))

    # Keep PPO's simple xy controller. The G1 mesh hierarchy supplies the
    # realistic robot appearance without introducing humanoid locomotion.
    _remove_xml_children(g1_body, {"freejoint", "joint", "inertial", "site"})
    g1_body.set("pos", "0 0 0.793")
    for geom in g1_body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("density", "0")

    return (
        ET.tostring(default, encoding="unicode"),
        "\n".join(ET.tostring(child, encoding="unicode") for child in list(asset)),
        ET.tostring(g1_body, encoding="unicode"),
    )


def _prepare_humanoid_body_xml() -> str:
    """Load humanoid body structure for the survivor target."""
    root = ET.parse(HUMANOID_MODEL_PATH).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Invalid humanoid MJCF: {HUMANOID_MODEL_PATH}")

    torso_body = worldbody.find("body[@name='torso']")
    if torso_body is None:
        raise ValueError(f"Humanoid torso body not found: {HUMANOID_MODEL_PATH}")

    for geom in torso_body.iter("geom"):
        geom.set("rgba", "1.0 0.25 0.25 1")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    return ET.tostring(torso_body, encoding="unicode")


def _burial_cover_xml(scene: dict, survivor: list[float]) -> str:
    survivor_meta = scene.get("survivor", {})
    if not survivor_meta.get("buried"):
        return ""

    sx, sy, _ = survivor
    cover = survivor_meta.get("cover", "rubble")
    rubble_pieces = [
        (-0.34, -0.12, 0.32, 0.16, 0.18, 0.18, "0.34 0.31 0.28 1"),
        (0.28, 0.14, 0.26, 0.20, 0.14, 0.18, "0.42 0.38 0.34 1"),
        (-0.02, 0.38, 0.42, 0.12, 0.12, 0.12, "0.48 0.44 0.39 1"),
        (0.16, -0.36, 0.18, 0.36, 0.12, 0.15, "0.29 0.27 0.25 1"),
    ]

    cover_xml = f'    <site name="buried_survivor_signal" pos="{sx} {sy} 0.08" size="0.16" rgba="0.0 0.85 1.0 0.9"/>\n'
    for i, (dx, dy, hx, hy, hz, z, rgba) in enumerate(rubble_pieces):
        cover_xml += (
            f'    <geom name="{cover}_cover_{i}" type="box" '
            f'pos="{sx + dx} {sy + dy} {z}" size="{hx} {hy} {hz}" '
            f'rgba="{rgba}" contype="0" conaffinity="0"/>\n'
        )
    return cover_xml


def _build_xml(scene: dict) -> str:
    survivor = _vec3(scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"]), default_z=0.0)
    obstacles = scene.get("obstacles", DEFAULT_SCENE["obstacles"])
    hazards = scene.get("hazards", DEFAULT_SCENE["hazards"])
    terrain = _normalize_terrain(scene.get("terrain"), scene.get("difficulty", "medium"))
    g1_default_xml, g1_asset_xml, g1_body_xml = _prepare_g1_robot_xml()
    humanoid_body_xml = _prepare_humanoid_body_xml()
    burial_cover_xml = _burial_cover_xml(scene, survivor)

    obs_xml = ""
    for i, obstacle in enumerate(obstacles):
        pos = _vec3(obstacle["pos"], default_z=1.0)
        size = list(obstacle.get("size", [0.5, 0.5, 1.0]))
        if len(size) == 2:
            size.append(1.0)
        sx, sy, sz = [float(v) for v in size[:3]]
        asset_id = obstacle.get("asset_id")
        rgba = ASSET_RGBA.get(asset_id, "0.55 0.27 0.07 1")
        yaw = math.radians(float(obstacle.get("rotation_yaw", 0.0)))
        name = _xml_name("obs", i, asset_id)
        obs_xml += (
            f'    <geom name="{html.escape(name)}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'euler="0 0 {yaw}" size="{sx} {sy} {sz}" rgba="{rgba}" '
            f'contype="0" conaffinity="0"/>\n'
        )

    haz_xml = ""
    for i, hazard in enumerate(hazards):
        center = _vec3(hazard.get("center", [0.0, 0.0, 0.0]), default_z=0.0)
        radius = float(hazard.get("radius", 1.0))
        height = float(hazard.get("height", 0.02))
        rgba = HAZARD_RGBA.get(hazard.get("type"), "1.0 0.15 0.15 0.35")
        name = _xml_name("haz", i, hazard.get("type"))
        haz_xml += (
            f'    <geom name="{html.escape(name)}" type="cylinder" pos="{center[0]} {center[1]} {height / 2}" '
            f'size="{radius} {height / 2}" rgba="{rgba}" '
            f'contype="0" conaffinity="0"/>\n'
        )

    inactive_survivor_xml = ""
    for i, survivor_info in enumerate(scene.get("survivors", [])):
        if survivor_info.get("active"):
            continue
        pos = _vec3(survivor_info.get("pos", [0.0, 0.0, 0.0]), default_z=0.0)
        name = _xml_name("inactive_survivor", i, survivor_info.get("name"))
        inactive_survivor_xml += (
            f'    <geom name="{html.escape(name)}" type="cylinder" '
            f'pos="{pos[0]} {pos[1]} 0.04" size="0.35 0.04" '
            f'rgba="1.0 0.7 0.2 0.55" contype="0" conaffinity="0"/>\n'
        )

    terrain_xml = _build_terrain_xml(terrain)

    return f"""
<mujoco model="grounded_disaster_rescue">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 0" integrator="RK4"/>

  {g1_default_xml}

  <asset>
    <texture name="checker" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.65 0.65 0.65" rgb2="0.45 0.45 0.45"/>
    <material name="grid_mat" texture="checker" texrepeat="8 8" reflectance="0.05"/>
{g1_asset_xml}
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="{WORLD_HALF} {WORLD_HALF} 0.1"
          pos="0 0 0" material="grid_mat" contype="1" conaffinity="1"/>
    <light name="sky" pos="0 0 15" dir="0 0 -1" diffuse="0.85 0.85 0.85" specular="0.2 0.2 0.2"/>
    <light name="front" pos="0 -12 8" dir="0 0.8 -0.6" diffuse="0.5 0.5 0.5"/>

{terrain_xml}
{obs_xml}
{haz_xml}
{burial_cover_xml}
{inactive_survivor_xml}

    <body name="survivor" pos="{survivor[0]} {survivor[1]} 0">
      {humanoid_body_xml}
      <geom name="survivor_ring" type="cylinder" size="0.55 0.02" pos="0 0 -0.55"
            rgba="1.0 0.9 0.0 0.8" contype="0" conaffinity="0"/>
    </body>

    <body name="robot" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>

      {g1_body_xml}
      <geom name="feet" type="cylinder" size="{ROBOT_RADIUS} 0.025" pos="0 0 0.025"
            rgba="0.0 0.9 0.45 0.8" contype="0" conaffinity="0"/>
      <camera name="track_robot" pos="0 -6 4" xyaxes="1 0 0 0 0.4 1" mode="track"/>
    </body>
  </worldbody>
</mujoco>
"""


def _normalize_terrain(terrain: dict | None, difficulty: str = "medium") -> dict:
    if not terrain:
        return generate_random_terrain(seed=None, difficulty=difficulty)
    if "heights" not in terrain or "roughness" not in terrain:
        return generate_random_terrain(
            seed=terrain.get("seed"),
            difficulty=difficulty,
            grid_size=int(terrain.get("grid_size", 10)),
        )
    return terrain


def _build_terrain_xml(terrain: dict) -> str:
    heights = terrain.get("heights", [])
    roughness = terrain.get("roughness", [])
    danger = terrain.get("danger", [])
    rigid = terrain.get("rigid", [])
    grid_size = int(terrain.get("grid_size") or len(heights) or 0)
    if grid_size <= 0:
        return ""

    cell = (WORLD_HALF * 2) / grid_size
    half = cell / 2.0
    xml = []
    for row in range(grid_size):
        for col in range(grid_size):
            height = float(heights[row][col])
            rough = float(roughness[row][col]) if roughness else 0.0
            danger_level = float(danger[row][col]) if danger else 0.0
            rigid_level = float(rigid[row][col]) if rigid else 0.0
            if height <= 0.01 and rough <= 0.05 and danger_level <= 0.0:
                continue
            x = -WORLD_HALF + half + col * cell
            y = -WORLD_HALF + half + row * cell
            slab_boost = 0.08 if rigid_level > 0 else 0.0
            z = max(0.006, (height + slab_boost) / 2.0)
            red = 0.32 + min(rough, 1.2) * 0.24 + danger_level * 0.18
            green = 0.36 + min(height, 0.75) * 0.28 - danger_level * 0.12
            blue = 0.32 + rigid_level * 0.14
            xml.append(
                f'    <geom name="terrain_{row}_{col}" type="box" pos="{x:.3f} {y:.3f} {z:.3f}" '
                f'size="{half:.3f} {half:.3f} {z:.3f}" rgba="{red:.3f} {green:.3f} {blue:.3f} 0.82" '
                f'contype="0" conaffinity="0"/>\n'
            )
    return "".join(xml)


def _terrain_cell_value(terrain: dict, xy: np.ndarray, key: str) -> float:
    values = terrain.get(key)
    grid_size = int(terrain.get("grid_size") or len(values or []) or 0)
    if not values or grid_size <= 0:
        return 0.0
    col = int(np.clip((xy[0] + WORLD_HALF) / (WORLD_HALF * 2) * grid_size, 0, grid_size - 1))
    row = int(np.clip((xy[1] + WORLD_HALF) / (WORLD_HALF * 2) * grid_size, 0, grid_size - 1))
    return float(values[row][col])


def _terrain_blocks_motion(terrain: dict, xy: np.ndarray) -> bool:
    height = _terrain_cell_value(terrain, xy, "heights")
    roughness = _terrain_cell_value(terrain, xy, "roughness")
    danger = _terrain_cell_value(terrain, xy, "danger")
    return height >= 0.58 or roughness >= 1.0 or danger >= 1.0


class DisasterEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, scene: dict | None = None, render_mode: str = "rgb_array"):
        super().__init__()
        self.scene = scene or DEFAULT_SCENE
        self.render_mode = render_mode
        self._step_count = 0
        self._prev_dist = None
        self._pos = _xy(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"]))
        self._prev_pos = self._pos.copy()
        self._surv_xy = _xy(self.scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"]))
        self._survivor_meta = dict(self.scene.get("survivor", {}))
        self._survivor_buried = bool(self._survivor_meta.get("buried", False))
        self._detection_radius = float(self._survivor_meta.get("detection_radius", 0.0))
        self._detected_once = False
        self._obstacles = [
            _obstacle_bounds(obstacle)
            for obstacle in self.scene.get("obstacles", DEFAULT_SCENE["obstacles"])
        ]
        self._hazards = [
            {
                "center": _xy(hazard.get("center", [0.0, 0.0, 0.0])),
                "radius": float(hazard.get("radius", 1.0)),
            }
            for hazard in self.scene.get("hazards", DEFAULT_SCENE["hazards"])
        ]
        self._terrain = _normalize_terrain(
            self.scene.get("terrain"),
            self.scene.get("difficulty", DEFAULT_SCENE["difficulty"]),
        )
        self.scene["terrain"] = self._terrain

        self._xml = _build_xml(self.scene)
        self._model = mujoco.MjModel.from_xml_string(self._xml)
        self._data = mujoco.MjData(self._model)
        self._jx_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
        self._jy_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
        self._renderer = None

        obs_high = np.array(
            [WORLD_HALF] * 2
            + [WORLD_HALF] * 2
            + [WORLD_HALF * 2] * 2
            + [WORLD_HALF * 2]
            + [ACTION_SPEED] * 2
            + [WORLD_HALF * 2, WORLD_HALF * 2, WORLD_HALF] * OBSTACLE_COUNT
            + [1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _sync_mujoco(self):
        qx = self._model.jnt_qposadr[self._jx_id]
        qy = self._model.jnt_qposadr[self._jy_id]
        self._data.qpos[qx] = self._pos[0]
        self._data.qpos[qy] = self._pos[1]
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _distance(self) -> float:
        return float(np.linalg.norm(self._surv_xy - self._pos))

    def _survivor_detected(self, dist: float | None = None) -> bool:
        if not self._survivor_buried or self._detection_radius <= 0.0:
            return False
        current_dist = self._distance() if dist is None else dist
        return current_dist <= self._detection_radius

    def _detection_info(self, dist: float | None = None) -> dict:
        detected = self._survivor_detected(dist)
        return {
            "survivor_buried": self._survivor_buried,
            "survivor_detected": detected,
            "detection_radius": self._detection_radius,
            "detection_signal": self._survivor_meta.get("signal"),
            "survivor_cover": self._survivor_meta.get("cover"),
        }

    def _detection_features(self, dist: float | None = None) -> list[float]:
        current_dist = self._distance() if dist is None else dist
        buried = 1.0 if self._survivor_buried else 0.0
        detected = 1.0 if self._survivor_detected(current_dist) else 0.0
        if not self._survivor_buried or self._detection_radius <= 0.0:
            strength = 0.0
        else:
            strength = max(0.0, 1.0 - current_dist / self._detection_radius)
        return [buried, detected, strength]

    def _collides(self, xy: np.ndarray) -> bool:
        for center, half in self._obstacles:
            expanded = half + ROBOT_RADIUS
            if np.all(np.abs(xy - center) <= expanded):
                return True
        return False

    def _in_hazard(self, xy: np.ndarray) -> bool:
        return any(np.linalg.norm(xy - h["center"]) < h["radius"] for h in self._hazards)

    def _obstacle_features(self) -> list[float]:
        features: list[float] = []
        feature_obstacles = sorted(
            self._obstacles,
            key=lambda obstacle: _obstacle_sort_key(self._pos, obstacle),
        )[:OBSTACLE_COUNT]
        for center, half in feature_obstacles:
            rel = center - self._pos
            outside = np.maximum(np.abs(self._pos - center) - (half + ROBOT_RADIUS), 0.0)
            clearance = float(np.linalg.norm(outside))
            if np.all(np.abs(self._pos - center) <= half + ROBOT_RADIUS):
                clearance = -float(np.min(half + ROBOT_RADIUS - np.abs(self._pos - center)))
            features.extend([rel[0], rel[1], clearance])
        while len(features) < OBSTACLE_COUNT * 3:
            features.extend([0.0, 0.0, WORLD_HALF])
        return features[:OBSTACLE_COUNT * 3]

    def _get_obs(self) -> np.ndarray:
        rel = self._surv_xy - self._pos
        dist = np.linalg.norm(rel)
        vel = self._pos - self._prev_pos
        return np.array(
            [
                *self._pos,
                *self._surv_xy,
                *rel,
                dist,
                *vel,
                *self._obstacle_features(),
                *self._detection_features(dist),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = _xy(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"]))
        self._prev_pos = self._pos.copy()
        self._step_count = 0
        self._prev_dist = self._distance()
        self._detected_once = self._survivor_detected(self._prev_dist)
        self._sync_mujoco()
        return self._get_obs(), self._detection_info(self._prev_dist)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm

        self._prev_pos = self._pos.copy()
        proposed = self._pos + np.clip(action, -1.0, 1.0) * ACTION_SPEED
        terrain_roughness = _terrain_cell_value(self._terrain, proposed, "roughness")
        terrain_danger = _terrain_cell_value(self._terrain, proposed, "danger")
        proposed = self._pos + (proposed - self._pos) * max(0.2, 1.0 - terrain_roughness * 0.58)
        proposed = np.clip(proposed, -WORLD_HALF + ROBOT_RADIUS, WORLD_HALF - ROBOT_RADIUS)

        terrain_blocked = _terrain_blocks_motion(self._terrain, proposed)
        collided = self._collides(proposed) or terrain_blocked
        if not collided:
            self._pos = proposed

        self._step_count += 1
        self._sync_mujoco()

        dist = self._distance()
        progress = 0.0 if self._prev_dist is None else self._prev_dist - dist
        self._prev_dist = dist

        reward = progress * 20.0 - dist * 0.01 - 0.02
        if collided:
            reward -= COLLISION_PEN
        if self._in_hazard(self._pos):
            reward -= HAZARD_PEN
        reward -= terrain_roughness * 0.45 + terrain_danger * 1.75
        if self._survivor_detected(dist) and not self._detected_once:
            reward += DETECTION_BONUS
            self._detected_once = True

        reached = dist < REACH_DIST
        truncated = self._step_count >= MAX_STEPS
        if reached:
            reward += REACH_BONUS

        info = {
            "reached": reached,
            "dist": dist,
            "steps": self._step_count,
            "collided": collided,
            "terrain_blocked": terrain_blocked,
            "z": GROUND_Z,
            "terrain_height": _terrain_cell_value(self._terrain, self._pos, "heights"),
            "terrain_roughness": terrain_roughness,
            "terrain_danger": terrain_danger,
            **self._detection_info(dist),
        }
        return self._get_obs(), float(reward), reached, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._renderer.update_scene(self._data, camera="track_robot")
            return self._renderer.render()
        if self.render_mode == "human":
            mujoco.viewer.launch(self._model, self._data)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
