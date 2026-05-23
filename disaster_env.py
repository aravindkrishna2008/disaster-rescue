"""
Grounded disaster rescue environment.

The rendered robot is humanoid-shaped, but the policy controls a grounded 2D
body. It cannot fly: there is no z action, no z joint, and the reward/distance
are computed only on the floor plane. Obstacles are visible MuJoCo boxes and
are enforced by Python collision checks so the policy must route around them.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

DEFAULT_SCENE = {
    "robot_start": [-4.0, -4.0, 0.0],
    "survivor_pos": [4.0, 4.0, 0.0],
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
MAX_STEPS = 600
WORLD_HALF = 8.0
ACTION_SPEED = 0.18
ROBOT_RADIUS = 0.35
GROUND_Z = 0.0
OBSTACLE_COUNT = 3


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


def _build_xml(scene: dict) -> str:
    survivor = _vec3(scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"]), default_z=0.0)
    obstacles = scene.get("obstacles", DEFAULT_SCENE["obstacles"])
    hazards = scene.get("hazards", DEFAULT_SCENE["hazards"])

    obs_xml = ""
    for i, obstacle in enumerate(obstacles):
        pos = _vec3(obstacle["pos"], default_z=1.0)
        size = list(obstacle.get("size", [0.5, 0.5, 1.0]))
        if len(size) == 2:
            size.append(1.0)
        sx, sy, sz = [float(v) for v in size[:3]]
        obs_xml += (
            f'    <geom name="obs_{i}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'size="{sx} {sy} {sz}" rgba="0.55 0.27 0.07 1" '
            f'contype="0" conaffinity="0"/>\n'
        )

    haz_xml = ""
    for i, hazard in enumerate(hazards):
        center = _vec3(hazard.get("center", [0.0, 0.0, 0.0]), default_z=0.0)
        radius = float(hazard.get("radius", 1.0))
        haz_xml += (
            f'    <geom name="haz_{i}" type="cylinder" pos="{center[0]} {center[1]} 0.01" '
            f'size="{radius} 0.01" rgba="1.0 0.15 0.15 0.35" '
            f'contype="0" conaffinity="0"/>\n'
        )

    return f"""
<mujoco model="grounded_disaster_rescue">
  <option timestep="0.02" gravity="0 0 0" integrator="RK4"/>

  <asset>
    <texture name="checker" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.65 0.65 0.65" rgb2="0.45 0.45 0.45"/>
    <material name="grid_mat" texture="checker" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="{WORLD_HALF} {WORLD_HALF} 0.1"
          pos="0 0 0" material="grid_mat" contype="1" conaffinity="1"/>
    <light name="sky" pos="0 0 15" dir="0 0 -1" diffuse="0.85 0.85 0.85" specular="0.2 0.2 0.2"/>
    <light name="front" pos="0 -12 8" dir="0 0.8 -0.6" diffuse="0.5 0.5 0.5"/>

{obs_xml}
{haz_xml}

    <body name="survivor" pos="{survivor[0]} {survivor[1]} 0.8">
      <geom name="survivor_body" type="capsule" size="0.18 0.35"
            rgba="1.0 0.25 0.25 1" contype="0" conaffinity="0"/>
      <geom name="survivor_ring" type="cylinder" size="0.55 0.02" pos="0 0 -0.55"
            rgba="1.0 0.9 0.0 0.8" contype="0" conaffinity="0"/>
    </body>

    <body name="robot" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>

      <geom name="torso" type="capsule" size="0.18 0.4" pos="0 0 0.55"
            rgba="0.1 0.45 1.0 1" mass="2" contype="0" conaffinity="0"/>
      <geom name="head" type="sphere" size="0.15" pos="0 0 1.0"
            rgba="0.2 0.5 1.0 1" mass="0.5" contype="0" conaffinity="0"/>
      <geom name="l_arm" type="capsule" size="0.08 0.3" pos="-0.25 0 0.7"
            rgba="0.15 0.4 0.95 1" mass="0.3" contype="0" conaffinity="0"/>
      <geom name="r_arm" type="capsule" size="0.08 0.3" pos="0.25 0 0.7"
            rgba="0.15 0.4 0.95 1" mass="0.3" contype="0" conaffinity="0"/>
      <geom name="l_leg" type="capsule" size="0.1 0.35" pos="-0.12 0 0.15"
            rgba="0.25 0.35 0.9 1" mass="0.4" contype="0" conaffinity="0"/>
      <geom name="r_leg" type="capsule" size="0.1 0.35" pos="0.12 0 0.15"
            rgba="0.25 0.35 0.9 1" mass="0.4" contype="0" conaffinity="0"/>
      <geom name="feet" type="cylinder" size="{ROBOT_RADIUS} 0.025" pos="0 0 0.025"
            rgba="0.0 0.9 0.45 0.8" contype="0" conaffinity="0"/>
      <camera name="track_robot" pos="0 -6 4" xyaxes="1 0 0 0 0.4 1" mode="track"/>
    </body>
  </worldbody>
</mujoco>
"""


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
        self._obstacles = [
            _obstacle_bounds(obstacle)
            for obstacle in self.scene.get("obstacles", DEFAULT_SCENE["obstacles"])[:OBSTACLE_COUNT]
        ]
        self._hazards = [
            {
                "center": _xy(hazard.get("center", [0.0, 0.0, 0.0])),
                "radius": float(hazard.get("radius", 1.0)),
            }
            for hazard in self.scene.get("hazards", DEFAULT_SCENE["hazards"])
        ]

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
            + [WORLD_HALF * 2, WORLD_HALF * 2, WORLD_HALF] * OBSTACLE_COUNT,
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
        for center, half in self._obstacles:
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
            [*self._pos, *self._surv_xy, *rel, dist, *vel, *self._obstacle_features()],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = _xy(self.scene.get("robot_start", DEFAULT_SCENE["robot_start"]))
        self._prev_pos = self._pos.copy()
        self._step_count = 0
        self._prev_dist = self._distance()
        self._sync_mujoco()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm

        self._prev_pos = self._pos.copy()
        proposed = self._pos + np.clip(action, -1.0, 1.0) * ACTION_SPEED
        proposed = np.clip(proposed, -WORLD_HALF + ROBOT_RADIUS, WORLD_HALF - ROBOT_RADIUS)

        collided = self._collides(proposed)
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

        reached = dist < REACH_DIST
        truncated = self._step_count >= MAX_STEPS
        if reached:
            reward += REACH_BONUS

        info = {
            "reached": reached,
            "dist": dist,
            "steps": self._step_count,
            "collided": collided,
            "z": GROUND_Z,
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
