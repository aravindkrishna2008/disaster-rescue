"""
Unitree G1 locomotion environment for disaster rescue.

This environment uses a real MuJoCo free-base robot under gravity. The policy
does not need to discover walking from scratch: a deterministic gait reference
and bounded target-directed assist provide the baseline locomotion, while PPO
learns residual joint corrections.
"""

from __future__ import annotations

import heapq
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

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

WORLD_HALF = 8.0
MAX_STEPS = 1_000
REACH_DIST = 0.9
REACH_BONUS = 300.0
FALL_PENALTY = 120.0
OBSTACLE_CONTACT_PENALTY = 45.0
HAZARD_STEP_PENALTY = 18.0
FALL_HEIGHT = 0.48
FALL_UPRIGHT_Z = 0.45
FRAME_SKIP = 10
GAIT_FREQUENCY = 1.35
GAIT_STEP_TIME = 0.005 * FRAME_SKIP
GAIT_ACTION_SCALE = 0.09
ACTION_SMOOTHING = 0.25
TARGET_SPEED = 1.0
ASSIST_FORCE_SCALE = 85.0
ASSIST_LATERAL_DAMPING = 18.0
ASSIST_YAW_TORQUE = 18.0
ASSIST_UPRIGHT_TORQUE = 95.0
ASSIST_ANGULAR_DAMPING = 8.0
ASSIST_HEIGHT_KP = 1_600.0
ASSIST_HEIGHT_KD = 180.0
ASSIST_VERTICAL_FORCE_LIMIT = 1_200.0
NAV_GRID_RES = 0.25
OBSTACLE_INFLATION = 1.0
HAZARD_INFLATION = 0.6
WAYPOINT_REACHED_DIST = 0.5
DEFAULT_BALANCE_ASSIST_SCALE = 0.85
STALLED_PROGRESS_EPS = 0.003
STANCE_SLIP_PENALTY = 0.8
SWING_CLEARANCE_REWARD = 0.25
ASSIST_FORCE_PENALTY = 0.0008
TORSO_SMOOTHNESS_PENALTY = 0.05
MIN_SWING_CLEARANCE = 0.055
OBSTACLE_COUNT = 8
TERRAIN_COUNT = 12
CURRICULUM_STAGES = (
    "stand",
    "walk",
    "target",
    "flat_walk",
    "low_assist_walk",
    "obstacle_nav",
    "natural_target",
)
STAGE_ACTION_SCALES = {
    "stand": 0.05,
    "flat_walk": 0.05,
    "walk": 0.07,
    "low_assist_walk": 0.065,
    "obstacle_nav": 0.08,
    "target": GAIT_ACTION_SCALE,
    "natural_target": GAIT_ACTION_SCALE,
}

G1_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "g1.xml"
)

CONTROLLED_ACTUATORS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

GAIT_OFFSETS = {
    "left_hip_pitch_joint": -0.24,
    "left_knee_joint": 0.26,
    "left_ankle_pitch_joint": -0.14,
    "right_hip_pitch_joint": 0.24,
    "right_knee_joint": -0.26,
    "right_ankle_pitch_joint": 0.14,
    "left_hip_roll_joint": 0.035,
    "right_hip_roll_joint": 0.035,
    "waist_yaw_joint": -0.06,
}

WALK_STAGES = {"walk", "flat_walk", "low_assist_walk"}
NAVIGATION_STAGES = {"target", "obstacle_nav", "natural_target"}

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


def _fmt_vec(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _xy(seq) -> np.ndarray:
    return np.array(_vec3(seq)[:2], dtype=np.float64)


def _obstacle_footprint(obstacle: dict) -> tuple[np.ndarray, np.ndarray]:
    center = _xy(obstacle.get("pos", [0.0, 0.0, 0.0]))
    size = list(obstacle.get("size", [0.5, 0.5, 1.0]))
    if len(size) == 1:
        size = [size[0], size[0], 1.0]
    if len(size) == 2:
        size.append(1.0)
    return center, np.array(size[:2], dtype=np.float64)


def _box_signed_clearance(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    delta = np.abs(point - center) - half
    outside = np.maximum(delta, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = float(np.minimum(np.max(delta), 0.0))
    return outside_dist + inside_dist


def _box_nearest_vector(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    nearest = np.clip(point, center - half, center + half)
    vec = nearest - point
    if np.linalg.norm(vec) > 1e-6:
        return vec
    direction = point - center
    if np.linalg.norm(direction) < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / np.linalg.norm(direction) * _box_signed_clearance(point, center, half)


def _hazard_signed_clearance(point: np.ndarray, hazard: dict) -> float:
    return float(np.linalg.norm(point - hazard["center"]) - hazard["radius"])


def _norm_or_zero(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec, dtype=np.float64)
    return vec / norm


def _add_scene_context(worldbody: ET.Element, scene: dict) -> None:
    for i, terrain in enumerate(scene.get("terrain", [])[:TERRAIN_COUNT]):
        pos = _vec3(terrain.get("pos", [0.0, 0.0, 0.03]), default_z=0.03)
        size = list(terrain.get("size", [0.6, 0.6, 0.03]))
        if len(size) == 2:
            size.append(0.03)
        attrs = {
            "name": f"scene_terrain_{i}",
            "type": terrain.get("type", "box"),
            "pos": _fmt_vec(pos),
            "size": _fmt_vec(size[:3]),
            "rgba": terrain.get("rgba", "0.34 0.36 0.32 1.0"),
            "contype": "1",
            "conaffinity": "1",
            "friction": terrain.get("friction", "1.15 0.01 0.0001"),
        }
        if "euler" in terrain:
            attrs["euler"] = _fmt_vec(terrain["euler"])
        ET.SubElement(worldbody, "geom", attrs)

    for i, obstacle in enumerate(scene.get("obstacles", DEFAULT_SCENE["obstacles"])[:OBSTACLE_COUNT]):
        pos = _vec3(obstacle.get("pos", [0.0, 0.0, 1.0]), default_z=1.0)
        size = list(obstacle.get("size", [0.5, 0.5, 1.0]))
        if len(size) == 2:
            size.append(1.0)
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"scene_obstacle_{i}",
                "type": "box",
                "pos": _fmt_vec(pos),
                "size": _fmt_vec(size[:3]),
                "rgba": "0.55 0.27 0.07 0.82",
                "contype": "1",
                "conaffinity": "1",
                "friction": "0.85 0.01 0.0001",
            },
        )

    for i, hazard in enumerate(scene.get("hazards", DEFAULT_SCENE["hazards"])):
        center = _vec3(hazard.get("center", [0.0, 0.0, 0.0]), default_z=0.0)
        radius = float(hazard.get("radius", 1.0))
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"scene_hazard_{i}",
                "type": "cylinder",
                "pos": _fmt_vec([center[0], center[1], 0.01]),
                "size": _fmt_vec([radius, 0.01]),
                "rgba": "1.0 0.15 0.15 0.28",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _build_xml(scene: dict) -> str:
    root = ET.parse(G1_MODEL_PATH).getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Invalid Unitree G1 MJCF: {G1_MODEL_PATH}")

    asset_dir = G1_MODEL_PATH.parent / "assets"
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh.set("file", str(asset_dir / mesh_file))

    ET.SubElement(
        asset,
        "texture",
        {
            "name": "rescue_grid",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": "0.64 0.64 0.64",
            "rgb2": "0.43 0.43 0.43",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "rescue_grid_mat",
            "texture": "rescue_grid",
            "texrepeat": "8 8",
            "reflectance": "0.04",
        },
    )

    ground = ET.Element(
        "geom",
        {
            "name": "rescue_ground",
            "type": "plane",
            "size": f"{WORLD_HALF} {WORLD_HALF} 0.1",
            "pos": "0 0 0",
            "material": "rescue_grid_mat",
            "friction": "1.0 0.005 0.0001",
        },
    )
    worldbody.insert(0, ground)

    survivor = _vec3(scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"]), default_z=0.0)
    target = ET.SubElement(
        worldbody,
        "body",
        {"name": "target_marker", "pos": _fmt_vec([survivor[0], survivor[1], 0.05])},
    )
    ET.SubElement(
        target,
        "geom",
        {
            "name": "target_ring",
            "type": "cylinder",
            "size": "0.65 0.025",
            "rgba": "1.0 0.9 0.0 0.8",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        target,
        "geom",
        {
            "name": "target_core",
            "type": "sphere",
            "pos": "0 0 0.65",
            "size": "0.16",
            "rgba": "1.0 0.15 0.15 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    _add_scene_context(worldbody, scene)

    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError(f"Unitree G1 pelvis body not found: {G1_MODEL_PATH}")
    if pelvis.find("camera[@name='track_robot']") is None:
        ET.SubElement(
            pelvis,
            "camera",
            {
                "name": "track_robot",
                "pos": "0 -6 3",
                "xyaxes": "1 0 0 0 0.45 1",
                "mode": "track",
            },
        )

    return ET.tostring(root, encoding="unicode")


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

    def __init__(
        self,
        scene: dict | None = None,
        render_mode: str = "rgb_array",
        curriculum_stage: str = "target",
        assist_enabled: bool = True,
        assist_scale: float | None = None,
        balance_assist_scale: float | None = None,
    ):
        super().__init__()
        if curriculum_stage not in CURRICULUM_STAGES:
            raise ValueError(f"Unknown curriculum_stage {curriculum_stage!r}")
        self.scene = scene or DEFAULT_SCENE
        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        self.assist_scale = float(1.0 if assist_scale is None and assist_enabled else assist_scale or 0.0)
        self.assist_scale = float(np.clip(self.assist_scale, 0.0, 1.0))
        if balance_assist_scale is None:
            balance_assist_scale = self.assist_scale
        self.balance_assist_scale = float(np.clip(balance_assist_scale, 0.0, 1.0))
        self.assist_enabled = self.assist_scale > 0.0 or self.balance_assist_scale > 0.0
        self._step_count = 0
        self._prev_dist = 0.0
        self._prev_guide_dist = 0.0
        self._prev_action = np.zeros(len(CONTROLLED_ACTUATORS), dtype=np.float32)
        self._gait_phase = 0.0
        self._renderer = None
        self._last_assist_force_norm = 0.0
        self._total_assist_force = 0.0
        self._total_stance_slip = 0.0
        self._total_swing_clearance = 0.0
        self._foot_metric_steps = 0
        self._obstacle_contact_count = 0
        self._hazard_step_count = 0
        self._last_foot_pos: dict[str, np.ndarray] = {}

        self._target_pos = np.array(
            _vec3(self.scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"]), default_z=0.0),
            dtype=np.float64,
        )
        self._target_pos[2] = 0.0
        self._obstacles = [
            _obstacle_footprint(obstacle)
            for obstacle in self.scene.get("obstacles", DEFAULT_SCENE["obstacles"])[:OBSTACLE_COUNT]
        ]
        self._hazards = [
            {
                "center": _xy(hazard.get("center", [0.0, 0.0, 0.0])),
                "radius": float(hazard.get("radius", 1.0)),
            }
            for hazard in self.scene.get("hazards", DEFAULT_SCENE["hazards"])
        ]
        self._path_waypoints = self._plan_path(
            _xy(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"])),
            self._target_pos[:2],
        )
        self._waypoint_index = 0

        self._xml = _build_xml(self.scene)
        self._model = mujoco.MjModel.from_xml_string(self._xml)
        self._data = mujoco.MjData(self._model)

        self._pelvis_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._torso_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._base_joint_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        if min(self._pelvis_body_id, self._torso_body_id, self._base_joint_id) < 0:
            raise ValueError("G1 model is missing pelvis, torso, or floating base joint")
        self._left_foot_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self._right_foot_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        if min(self._left_foot_site_id, self._right_foot_site_id) < 0:
            raise ValueError("G1 model is missing left_foot or right_foot sites")
        self._ground_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "rescue_ground")
        self._obstacle_geom_ids = {
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"scene_obstacle_{i}")
            for i in range(len(self._obstacles))
        }
        self._obstacle_geom_ids.discard(-1)
        self._terrain_geom_ids = {
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"scene_terrain_{i}")
            for i in range(len(self.scene.get("terrain", [])[:TERRAIN_COUNT]))
        }
        self._terrain_geom_ids.discard(-1)
        self._support_geom_ids = {self._ground_geom_id} | self._terrain_geom_ids

        self._base_qpos_adr = self._model.jnt_qposadr[self._base_joint_id]
        self._base_dof_adr = self._model.jnt_dofadr[self._base_joint_id]
        self._controlled_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in CONTROLLED_ACTUATORS
            ],
            dtype=np.int32,
        )
        if np.any(self._controlled_actuator_ids < 0):
            missing = [
                name
                for name, actuator_id in zip(CONTROLLED_ACTUATORS, self._controlled_actuator_ids)
                if actuator_id < 0
            ]
            raise ValueError(f"Missing G1 actuators: {missing}")

        self._controlled_joint_ids = self._model.actuator_trnid[self._controlled_actuator_ids, 0]
        self._controlled_qpos_adrs = self._model.jnt_qposadr[self._controlled_joint_ids]
        self._controlled_dof_adrs = self._model.jnt_dofadr[self._controlled_joint_ids]
        self._controlled_name_to_index = {
            name: idx for idx, name in enumerate(CONTROLLED_ACTUATORS)
        }
        self._left_foot_geom_ids = self._geom_ids_for_body("left_ankle_roll_link")
        self._right_foot_geom_ids = self._geom_ids_for_body("right_ankle_roll_link")
        self._foot_geom_ids = self._left_foot_geom_ids | self._right_foot_geom_ids

        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0:
            self._stand_qpos = np.zeros(self._model.nq, dtype=np.float64)
            self._stand_qpos[self._base_qpos_adr + 2] = 0.79
            self._stand_qpos[self._base_qpos_adr + 3] = 1.0
            self._stand_ctrl = np.zeros(self._model.nu, dtype=np.float64)
        else:
            self._stand_qpos = np.array(self._model.key_qpos[key_id], dtype=np.float64)
            self._stand_ctrl = np.array(self._model.key_ctrl[key_id], dtype=np.float64)

        self._ctrl_low = np.array(self._model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self._ctrl_high = np.array(self._model.actuator_ctrlrange[:, 1], dtype=np.float64)
        invalid_range = self._ctrl_low >= self._ctrl_high
        self._ctrl_low[invalid_range] = -np.pi
        self._ctrl_high[invalid_range] = np.pi

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(CONTROLLED_ACTUATORS),),
            dtype=np.float32,
        )

        self._reset_physics_state()
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=np.full(obs.shape, -100.0, dtype=np.float32),
            high=np.full(obs.shape, 100.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _geom_ids_for_body(self, body_name: str) -> set[int]:
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return set()
        return {geom_id for geom_id in range(self._model.ngeom) if self._model.geom_bodyid[geom_id] == body_id}

    def _is_blocked_xy(self, point: np.ndarray) -> bool:
        if np.any(point < -WORLD_HALF) or np.any(point > WORLD_HALF):
            return True
        for center, half in self._obstacles:
            if _box_signed_clearance(point, center, half + OBSTACLE_INFLATION) <= 0.0:
                return True
        for hazard in self._hazards:
            if _hazard_signed_clearance(point, hazard) <= HAZARD_INFLATION:
                return True
        return False

    def _nearest_free_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        max_index = int(round((WORLD_HALF * 2.0) / NAV_GRID_RES))
        if not self._is_blocked_xy(self._cell_to_world(cell)):
            return cell
        queue = [(0, cell)]
        seen = {cell}
        while queue:
            _, current = heapq.heappop(queue)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in seen or nxt[0] < 0 or nxt[1] < 0 or nxt[0] > max_index or nxt[1] > max_index:
                    continue
                seen.add(nxt)
                if not self._is_blocked_xy(self._cell_to_world(nxt)):
                    return nxt
                heapq.heappush(queue, (abs(nxt[0] - cell[0]) + abs(nxt[1] - cell[1]), nxt))
        return cell

    def _world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        clipped = np.clip(point, -WORLD_HALF, WORLD_HALF)
        return (
            int(round((clipped[0] + WORLD_HALF) / NAV_GRID_RES)),
            int(round((clipped[1] + WORLD_HALF) / NAV_GRID_RES)),
        )

    def _cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array(
            [
                cell[0] * NAV_GRID_RES - WORLD_HALF,
                cell[1] * NAV_GRID_RES - WORLD_HALF,
            ],
            dtype=np.float64,
        )

    def _plan_path(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        start_cell = self._nearest_free_cell(self._world_to_cell(start))
        goal_cell = self._nearest_free_cell(self._world_to_cell(goal))
        if start_cell == goal_cell:
            return [goal.astype(np.float64)]

        max_index = int(round((WORLD_HALF * 2.0) / NAV_GRID_RES))
        open_heap = [(0.0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start_cell: 0.0}

        def heuristic(cell: tuple[int, int]) -> float:
            return float(np.linalg.norm(np.array(cell, dtype=np.float64) - np.array(goal_cell, dtype=np.float64)))

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                waypoints = [self._cell_to_world(cell) for cell in path[:: max(1, int(0.75 / NAV_GRID_RES))]]
                if np.linalg.norm(waypoints[-1] - goal) > 1e-6:
                    waypoints.append(goal.astype(np.float64))
                return waypoints

            for dx, dy, cost in (
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, 1.414),
                (1, -1, 1.414),
                (-1, 1, 1.414),
                (-1, -1, 1.414),
            ):
                nxt = (current[0] + dx, current[1] + dy)
                if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > max_index or nxt[1] > max_index:
                    continue
                if self._is_blocked_xy(self._cell_to_world(nxt)):
                    continue
                tentative = g_score[current] + cost
                if tentative >= g_score.get(nxt, np.inf):
                    continue
                came_from[nxt] = current
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))

        return [goal.astype(np.float64)]

    def _reset_physics_state(self) -> None:
        self._data.qpos[:] = self._stand_qpos
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = self._stand_ctrl
        self._data.xfrc_applied[:] = 0.0

        start = _vec3(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"]), default_z=0.0)
        self._data.qpos[self._base_qpos_adr : self._base_qpos_adr + 2] = start[:2]
        self._data.qpos[self._base_qpos_adr + 2] = self._stand_qpos[self._base_qpos_adr + 2]

        self._step_count = 0
        self._gait_phase = 0.0
        self._prev_action[:] = 0.0
        self._waypoint_index = 0
        self._last_assist_force_norm = 0.0
        self._total_assist_force = 0.0
        self._total_stance_slip = 0.0
        self._total_swing_clearance = 0.0
        self._foot_metric_steps = 0
        self._obstacle_contact_count = 0
        self._hazard_step_count = 0
        mujoco.mj_forward(self._model, self._data)
        self._prev_dist = self._target_distance()
        self._prev_guide_dist = self._guide_distance()
        self._last_foot_pos = {
            "left": self._foot_site_pos("left"),
            "right": self._foot_site_pos("right"),
        }

    def _base_pos(self) -> np.ndarray:
        return np.array(self._data.qpos[self._base_qpos_adr : self._base_qpos_adr + 3], dtype=np.float64)

    def _base_quat(self) -> np.ndarray:
        return np.array(self._data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7], dtype=np.float64)

    def _base_vel(self) -> np.ndarray:
        return np.array(self._data.qvel[self._base_dof_adr : self._base_dof_adr + 6], dtype=np.float64)

    def _torso_upright(self) -> float:
        xmat = self._data.xmat[self._torso_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _base_yaw(self) -> float:
        return _yaw_from_quat(self._base_quat())

    def _stage_index(self) -> int:
        return CURRICULUM_STAGES.index(self.curriculum_stage)

    def _desired_speed(self) -> float:
        if self.curriculum_stage == "stand":
            return 0.0
        if self.curriculum_stage in WALK_STAGES:
            return TARGET_SPEED * 0.75
        return TARGET_SPEED

    def _action_scale(self) -> float:
        return STAGE_ACTION_SCALES[self.curriculum_stage]

    def _effective_target_pos(self) -> np.ndarray:
        if self.curriculum_stage == "stand":
            target = self._base_pos().copy()
            target[2] = 0.0
            return target
        if self.curriculum_stage in WALK_STAGES:
            start = _vec3(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"]), default_z=0.0)
            return np.array([start[0] + 5.0, start[1], 0.0], dtype=np.float64)
        return self._target_pos

    def _active_waypoint_xy(self) -> np.ndarray:
        if self.curriculum_stage not in NAVIGATION_STAGES or not self._path_waypoints:
            return self._effective_target_pos()[:2]
        base_xy = self._base_pos()[:2]
        while self._waypoint_index < len(self._path_waypoints) - 1:
            if np.linalg.norm(self._path_waypoints[self._waypoint_index] - base_xy) > WAYPOINT_REACHED_DIST:
                break
            self._waypoint_index += 1
        return self._path_waypoints[self._waypoint_index]

    def _guide_pos(self) -> np.ndarray:
        guide = np.zeros(3, dtype=np.float64)
        guide[:2] = self._active_waypoint_xy()
        return guide

    def _guide_vector(self) -> np.ndarray:
        vec = self._active_waypoint_xy() - self._base_pos()[:2]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vec / norm

    def _target_vector(self) -> np.ndarray:
        vec = self._effective_target_pos()[:2] - self._base_pos()[:2]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vec / norm

    def _heading_error(self) -> float:
        direction = self._guide_vector()
        desired_yaw = float(np.arctan2(direction[1], direction[0]))
        return _wrap_angle(desired_yaw - self._base_yaw())

    def _target_distance(self) -> float:
        return float(np.linalg.norm(self._effective_target_pos()[:2] - self._base_pos()[:2]))

    def _guide_distance(self) -> float:
        return float(np.linalg.norm(self._active_waypoint_xy() - self._base_pos()[:2]))

    def _is_alive(self) -> bool:
        base_pos = self._base_pos()
        return (
            bool(np.all(np.isfinite(self._data.qpos)))
            and bool(np.all(np.isfinite(self._data.qvel)))
            and base_pos[2] > FALL_HEIGHT
            and self._torso_upright() > FALL_UPRIGHT_Z
        )

    def _foot_site_pos(self, side: str) -> np.ndarray:
        site_id = self._left_foot_site_id if side == "left" else self._right_foot_site_id
        return np.array(self._data.site_xpos[site_id], dtype=np.float64)

    def _foot_contacts(self) -> tuple[bool, bool]:
        left_contact = False
        right_contact = False
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geoms = {int(contact.geom1), int(contact.geom2)}
            if not (geoms & self._support_geom_ids):
                continue
            if geoms & self._left_foot_geom_ids:
                left_contact = True
            if geoms & self._right_foot_geom_ids:
                right_contact = True
        return left_contact, right_contact

    def _obstacle_contacts_this_step(self) -> int:
        contacts = 0
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            if int(contact.geom1) in self._obstacle_geom_ids or int(contact.geom2) in self._obstacle_geom_ids:
                contacts += 1
        return contacts

    def _clearance_metrics(self) -> tuple[float, float]:
        point = self._base_pos()[:2]
        obstacle_clearance = min(
            (
                _box_signed_clearance(point, center, half)
                for center, half in self._obstacles
            ),
            default=WORLD_HALF,
        )
        hazard_clearance = min(
            (_hazard_signed_clearance(point, hazard) for hazard in self._hazards),
            default=WORLD_HALF,
        )
        return float(obstacle_clearance), float(hazard_clearance)

    def _navigation_features(self) -> np.ndarray:
        base_xy = self._base_pos()[:2]
        guide_rel = self._guide_pos()[:2] - base_xy
        waypoint_remaining = max(len(self._path_waypoints) - 1 - self._waypoint_index, 0)

        nearest_obs_vec = np.zeros(2, dtype=np.float64)
        nearest_obs_clearance = WORLD_HALF
        for center, half in self._obstacles:
            clearance = _box_signed_clearance(base_xy, center, half)
            if clearance < nearest_obs_clearance:
                nearest_obs_clearance = clearance
                nearest_obs_vec = _box_nearest_vector(base_xy, center, half)

        nearest_hazard_vec = np.zeros(2, dtype=np.float64)
        nearest_hazard_clearance = WORLD_HALF
        for hazard in self._hazards:
            vec = hazard["center"] - base_xy
            clearance = _hazard_signed_clearance(base_xy, hazard)
            if clearance < nearest_hazard_clearance:
                nearest_hazard_clearance = clearance
                nearest_hazard_vec = vec

        return np.array(
            [
                guide_rel[0],
                guide_rel[1],
                np.linalg.norm(guide_rel),
                float(waypoint_remaining),
                nearest_obs_vec[0],
                nearest_obs_vec[1],
                nearest_obs_clearance,
                nearest_hazard_vec[0],
                nearest_hazard_vec[1],
                nearest_hazard_clearance,
            ],
            dtype=np.float64,
        )

    def _foot_features(self) -> np.ndarray:
        left_contact, right_contact = self._foot_contacts()
        left = self._foot_site_pos("left")
        right = self._foot_site_pos("right")
        base = self._base_pos()
        return np.array(
            [
                float(left_contact),
                float(right_contact),
                left[2],
                right[2],
                left[0] - base[0],
                left[1] - base[1],
                right[0] - base[0],
                right[1] - base[1],
            ],
            dtype=np.float64,
        )

    def _step_foot_metrics(self) -> dict:
        left_contact, right_contact = self._foot_contacts()
        left = self._foot_site_pos("left")
        right = self._foot_site_pos("right")
        dt = float(self._model.opt.timestep) * FRAME_SKIP

        stance_slip = 0.0
        contact_count = 0
        for side, pos, contact in (("left", left, left_contact), ("right", right, right_contact)):
            prev = self._last_foot_pos.get(side, pos)
            horizontal_speed = float(np.linalg.norm(pos[:2] - prev[:2]) / max(dt, 1e-6))
            if contact:
                stance_slip += horizontal_speed
                contact_count += 1
            self._last_foot_pos[side] = pos

        stance_slip = stance_slip / max(contact_count, 1)
        swing_clearance = 0.0
        swing_count = 0
        if not left_contact:
            swing_clearance += max(float(left[2]), 0.0)
            swing_count += 1
        if not right_contact:
            swing_clearance += max(float(right[2]), 0.0)
            swing_count += 1
        swing_clearance = swing_clearance / max(swing_count, 1)

        expected_left_swing = np.sin(self._gait_phase) > 0.0
        contact_match = 0.0
        if expected_left_swing:
            contact_match = float((not left_contact) and right_contact)
        else:
            contact_match = float(left_contact and (not right_contact))

        self._total_stance_slip += stance_slip
        self._total_swing_clearance += swing_clearance
        self._foot_metric_steps += 1
        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "stance_slip": stance_slip,
            "swing_clearance": swing_clearance,
            "contact_match": contact_match,
        }

    def _gait_reference_offsets(self) -> np.ndarray:
        offsets = np.zeros(len(CONTROLLED_ACTUATORS), dtype=np.float64)
        if self.curriculum_stage == "stand":
            return offsets

        phase = self._gait_phase
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        left_swing = max(sin_phase, 0.0)
        right_swing = max(-sin_phase, 0.0)
        left_stance = max(-sin_phase, 0.0)
        right_stance = max(sin_phase, 0.0)

        values = {
            "left_hip_pitch_joint": -0.11 * left_swing + 0.055 * left_stance,
            "right_hip_pitch_joint": -0.11 * right_swing + 0.055 * right_stance,
            "left_knee_joint": 0.18 * left_swing + 0.025 * left_stance,
            "right_knee_joint": 0.18 * right_swing + 0.025 * right_stance,
            "left_ankle_pitch_joint": -0.10 * left_swing - 0.025 * left_stance,
            "right_ankle_pitch_joint": -0.10 * right_swing - 0.025 * right_stance,
            "left_hip_roll_joint": 0.015 * cos_phase,
            "right_hip_roll_joint": -0.015 * cos_phase,
            "waist_yaw_joint": 0.02 * sin_phase,
        }
        for name, value in values.items():
            idx = self._controlled_name_to_index.get(name)
            if idx is not None:
                offsets[idx] = value

        heading_error = self._heading_error()
        waist_idx = self._controlled_name_to_index.get("waist_yaw_joint")
        if waist_idx is not None:
            offsets[waist_idx] += np.clip(heading_error, -0.45, 0.45) * 0.12

        return offsets

    def _apply_assist_forces(self) -> None:
        self._data.xfrc_applied[:] = 0.0
        self._last_assist_force_norm = 0.0
        if (
            (self.assist_scale <= 0.0 and self.balance_assist_scale <= 0.0)
            or self.curriculum_stage == "stand"
        ):
            return

        direction = self._guide_vector()
        base_vel = self._base_vel()
        xy_vel = base_vel[:2]
        speed_along = float(np.dot(xy_vel, direction))
        lateral_vel = xy_vel - speed_along * direction
        speed_error = np.clip(self._desired_speed() - speed_along, -1.0, 1.0)
        force_xy = ASSIST_FORCE_SCALE * speed_error * direction - ASSIST_LATERAL_DAMPING * lateral_vel
        height_error = self._stand_qpos[self._base_qpos_adr + 2] - self._base_pos()[2]
        force_z = np.clip(
            ASSIST_HEIGHT_KP * height_error - ASSIST_HEIGHT_KD * base_vel[2],
            0.0,
            ASSIST_VERTICAL_FORCE_LIMIT,
        )

        torso_mat = self._data.xmat[self._torso_body_id].reshape(3, 3)
        body_up = torso_mat[:, 2]
        upright_torque = np.cross(body_up, np.array([0.0, 0.0, 1.0])) * ASSIST_UPRIGHT_TORQUE
        heading_torque = np.array([0.0, 0.0, self._heading_error() * ASSIST_YAW_TORQUE])
        angular_damping = -ASSIST_ANGULAR_DAMPING * base_vel[3:6]
        torque = upright_torque + heading_torque + angular_damping

        force = np.array(
            [
                force_xy[0] * self.assist_scale,
                force_xy[1] * self.assist_scale,
                force_z * self.balance_assist_scale,
            ],
            dtype=np.float64,
        )
        torque = torque * self.balance_assist_scale
        self._last_assist_force_norm = float(np.linalg.norm(force) + 0.05 * np.linalg.norm(torque))
        self._data.xfrc_applied[self._pelvis_body_id, :3] = force
        self._data.xfrc_applied[self._pelvis_body_id, 3:] = torque

    def _get_obs(self) -> np.ndarray:
        base_pos = self._base_pos()
        base_quat = self._base_quat()
        base_vel = self._base_vel()
        joint_pos = self._data.qpos[self._controlled_qpos_adrs] - self._stand_qpos[self._controlled_qpos_adrs]
        joint_vel = self._data.qvel[self._controlled_dof_adrs]
        target_rel = self._effective_target_pos() - base_pos
        guide_rel = self._guide_pos() - base_pos
        dist = np.array([self._target_distance()], dtype=np.float64)
        body_features = np.array([base_pos[2], self._torso_upright()], dtype=np.float64)
        sensors = np.array(self._data.sensordata, dtype=np.float64)
        direction = self._guide_vector()
        gait_features = np.array(
            [
                np.sin(self._gait_phase),
                np.cos(self._gait_phase),
                direction[0],
                direction[1],
                self._heading_error(),
                self._desired_speed(),
                self.assist_scale,
            ],
            dtype=np.float64,
        )
        stage_one_hot = np.zeros(len(CURRICULUM_STAGES), dtype=np.float64)
        stage_one_hot[self._stage_index()] = 1.0

        obs = np.concatenate(
            [
                base_pos,
                base_quat,
                base_vel,
                joint_pos,
                joint_vel,
                target_rel,
                guide_rel,
                dist,
                body_features,
                gait_features,
                stage_one_hot,
                self._navigation_features(),
                self._foot_features(),
                sensors,
                self._prev_action.astype(np.float64),
            ]
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_physics_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        residual_action = (
            (1.0 - ACTION_SMOOTHING) * self._prev_action.astype(np.float64)
            + ACTION_SMOOTHING * action
        )
        gait_offsets = self._gait_reference_offsets()
        self._data.ctrl[:] = self._stand_ctrl
        target_ctrl = (
            self._stand_ctrl[self._controlled_actuator_ids]
            + gait_offsets
            + residual_action * self._action_scale()
        )
        target_ctrl = np.clip(
            target_ctrl,
            self._ctrl_low[self._controlled_actuator_ids],
            self._ctrl_high[self._controlled_actuator_ids],
        )
        self._data.ctrl[self._controlled_actuator_ids] = target_ctrl

        prev_dist = self._target_distance()
        prev_guide_dist = self._guide_distance()
        prev_waypoint_index = self._waypoint_index
        assist_force_sum = 0.0
        for _ in range(FRAME_SKIP):
            self._apply_assist_forces()
            assist_force_sum += self._last_assist_force_norm
            mujoco.mj_step(self._model, self._data)
        self._data.xfrc_applied[:] = 0.0
        mean_assist_force = assist_force_sum / FRAME_SKIP
        self._total_assist_force += mean_assist_force

        self._step_count += 1
        self._gait_phase = (self._gait_phase + 2.0 * np.pi * GAIT_FREQUENCY * GAIT_STEP_TIME) % (2.0 * np.pi)
        self._prev_action = residual_action.astype(np.float32)

        dist = self._target_distance()
        guide_dist = self._guide_distance()
        progress = prev_dist - dist
        guide_progress = prev_guide_dist - guide_dist
        waypoints_advanced = max(self._waypoint_index - prev_waypoint_index, 0)
        upright = self._torso_upright()
        alive = self._is_alive()
        reached = alive and self.curriculum_stage in NAVIGATION_STAGES and dist < REACH_DIST
        terminated = reached or not alive
        truncated = self._step_count >= MAX_STEPS
        obstacle_contacts = self._obstacle_contacts_this_step()
        self._obstacle_contact_count += obstacle_contacts
        obstacle_clearance, hazard_clearance = self._clearance_metrics()
        in_hazard = hazard_clearance < 0.0
        if in_hazard:
            self._hazard_step_count += 1
        foot_metrics = self._step_foot_metrics()

        control_penalty = 0.002 * float(np.sum(np.square(residual_action)))
        joint_vel_penalty = 0.0005 * float(np.sum(np.square(self._data.qvel[self._controlled_dof_adrs])))
        base_vel = self._base_vel()
        direction = self._guide_vector()
        speed_along = float(np.dot(base_vel[:2], direction))
        speed_reward = min(max(speed_along, 0.0), self._desired_speed() + 0.4)
        heading_reward = max(0.0, np.cos(self._heading_error()))
        swing_clearance_reward = min(foot_metrics["swing_clearance"], MIN_SWING_CLEARANCE) / MIN_SWING_CLEARANCE
        torso_smoothness = float(np.linalg.norm(base_vel[3:5]))
        alive_reward = 1.25 if self.curriculum_stage == "stand" else 0.08
        upright_reward = 1.4 if self.curriculum_stage == "stand" else 0.35
        stall_penalty = 0.0
        if self.curriculum_stage != "stand" and alive and not reached:
            if guide_progress < STALLED_PROGRESS_EPS and speed_along < self._desired_speed() * 0.2:
                stall_penalty += 0.35
            if guide_progress < -STALLED_PROGRESS_EPS:
                stall_penalty += min(abs(guide_progress) * 4.0, 0.5)
        reward = (
            np.clip(progress, -0.05, 0.25) * 30.0
            + np.clip(guide_progress, -0.05, 0.25) * 80.0
            + waypoints_advanced * 8.0
            + alive_reward
            + max(upright, 0.0) * upright_reward
            + speed_reward * 1.2
            + heading_reward * 0.15
            + foot_metrics["contact_match"] * 0.25
            + swing_clearance_reward * SWING_CLEARANCE_REWARD
            - guide_dist * 0.06
            - stall_penalty
            - control_penalty
            - joint_vel_penalty
            - foot_metrics["stance_slip"] * STANCE_SLIP_PENALTY
            - mean_assist_force * ASSIST_FORCE_PENALTY
            - torso_smoothness * TORSO_SMOOTHNESS_PENALTY
        )
        if self.curriculum_stage == "stand":
            reward += max(upright, 0.0) * 1.2 - 0.2 * float(np.linalg.norm(base_vel[:2]))
        if obstacle_contacts:
            reward -= OBSTACLE_CONTACT_PENALTY * obstacle_contacts
        if in_hazard:
            reward -= HAZARD_STEP_PENALTY
        if reached:
            reward += REACH_BONUS
        if not alive:
            reward -= FALL_PENALTY

        info = {
            "reached": reached,
            "dist": dist,
            "dist_3d": float(np.linalg.norm(self._effective_target_pos() - self._base_pos())),
            "steps": self._step_count,
            "fallen": not alive,
            "upright": upright,
            "base_pos": self._base_pos().astype(float).tolist(),
            "target_pos": self._effective_target_pos().astype(float).tolist(),
            "guide_pos": self._guide_pos().astype(float).tolist(),
            "guide_dist": guide_dist,
            "guide_progress": guide_progress,
            "waypoints_advanced": waypoints_advanced,
            "heading_error": self._heading_error(),
            "gait_phase": self._gait_phase,
            "curriculum_stage": self.curriculum_stage,
            "assist_enabled": self.assist_enabled,
            "assist_scale": self.assist_scale,
            "balance_assist_scale": self.balance_assist_scale,
            "mean_assist_force": mean_assist_force,
            "mean_assist_force_episode": self._total_assist_force / max(self._step_count, 1),
            "obstacle_contacts": self._obstacle_contact_count,
            "obstacle_contacts_step": obstacle_contacts,
            "min_obstacle_clearance": obstacle_clearance,
            "hazard_steps": self._hazard_step_count,
            "in_hazard": in_hazard,
            "min_hazard_clearance": hazard_clearance,
            "left_foot_contact": foot_metrics["left_contact"],
            "right_foot_contact": foot_metrics["right_contact"],
            "stance_slip": foot_metrics["stance_slip"],
            "mean_stance_slip": self._total_stance_slip / max(self._foot_metric_steps, 1),
            "swing_clearance": foot_metrics["swing_clearance"],
            "mean_swing_clearance": self._total_swing_clearance / max(self._foot_metric_steps, 1),
            "gait_score": float(
                np.clip(
                    1.0
                    - self._total_stance_slip / max(self._foot_metric_steps, 1)
                    + 0.5 * self._total_swing_clearance / max(self._foot_metric_steps, 1),
                    0.0,
                    1.0,
                )
            ),
            "z": float(self._base_pos()[2]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

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
