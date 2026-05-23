"""
3D Disaster Rescue Environment — MuJoCo-based Gymnasium env.

The robot is a sphere that navigates in full 3D (x, y, z) toward a survivor,
avoiding obstacles (solid boxes) and hazards (penalty zones on the floor).
Gravity is disabled so the robot acts like a drone/UAV in 3D space.

Observation (13,): robot_xyz(3) + survivor_xyz(3) + relative_xyz(3) + dist(1) + robot_vel(3)
Action      (3,):   velocity commands in x, y, z  ∈ [-1, 1]
"""

import os
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

# ── helpers ──────────────────────────────────────────────────────────────────

DEFAULT_SCENE = {
    "robot_start": [-4.0, -4.0, 1.0],
    "survivor_pos": [4.0, 4.0, 1.0],
    "obstacles": [
        {"pos": [0.0, 0.0, 1.0],  "size": [0.5, 0.5, 1.0]},
        {"pos": [2.0, -2.0, 0.75], "size": [0.4, 0.4, 0.75]},
        {"pos": [-2.0, 2.0, 1.25], "size": [0.4, 0.4, 1.25]},
    ],
    "hazards": [
        {"center": [1.0, 1.0, 0.0], "radius": 1.2},
        {"center": [-1.5, -1.0, 0.0], "radius": 0.9},
    ],
    "difficulty": "medium",
}

REACH_DIST  = 1.0   # metres — survivor reached
HAZARD_PEN  = 2.0   # reward penalty per step inside a hazard
REACH_BONUS = 150.0
MAX_STEPS   = 600
WORLD_HALF  = 8.0   # bounds — robot is penalised outside
ACTION_SPEED = 2.5

# ── MJCF builder ─────────────────────────────────────────────────────────────

def _vec3(seq, default_z=1.0):
    """Accept 2-D or 3-D list from Gemini; always return list of 3 floats."""
    seq = list(seq)
    if len(seq) == 2:
        seq.append(default_z)
    return [float(v) for v in seq[:3]]


def _build_xml(scene: dict) -> str:
    robot = _vec3(scene.get("robot_start", DEFAULT_SCENE["robot_start"]))
    surv  = _vec3(scene.get("survivor_pos",  DEFAULT_SCENE["survivor_pos"]))
    obs   = scene.get("obstacles", DEFAULT_SCENE["obstacles"])
    haz   = scene.get("hazards",   DEFAULT_SCENE["hazards"])

    # ── obstacle geoms ──
    obs_xml = ""
    for i, o in enumerate(obs):
        p = _vec3(o["pos"], default_z=1.0)
        s = list(o.get("size", [0.5, 0.5, 1.0]))
        if len(s) == 2:
            s.append(1.0)
        s = [float(v) for v in s[:3]]
        obs_xml += (
            f'    <geom name="obs_{i}" type="box" '
            f'pos="{p[0]} {p[1]} {p[2]}" '
            f'size="{s[0]} {s[1]} {s[2]}" '
            f'rgba="0.6 0.3 0.1 1" contype="0" conaffinity="0"/>\n'
        )

    # ── hazard floor markers (visual only — collision handled in Python) ──
    haz_xml = ""
    for i, h in enumerate(haz):
        c = _vec3(h.get("center", [0, 0, 0]), default_z=0.0)
        r = float(h.get("radius", 1.0))
        haz_xml += (
            f'    <geom name="haz_{i}" type="cylinder" '
            f'pos="{c[0]} {c[1]} 0.01" size="{r} 0.01" '
            f'rgba="1.0 0.2 0.2 0.35" contype="0" conaffinity="0"/>\n'
        )

    xml = f"""
<mujoco model="disaster_rescue">
  <option timestep="0.02" gravity="0 0 0" integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.7 0.7 0.7" rgb2="0.5 0.5 0.5"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="{WORLD_HALF} {WORLD_HALF} 0.1"
          pos="0 0 0" material="grid_mat" contype="1" conaffinity="1"/>

    <!-- Ambient + directional lighting -->
    <light name="top"   pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light name="front" pos="0 -10 6" dir="0 1 -0.5" diffuse="0.5 0.5 0.5"/>

    <!-- Obstacles -->
{obs_xml}
    <!-- Hazard zone markers -->
{haz_xml}
    <!-- Survivor (static capsule) -->
    <body name="survivor" pos="{surv[0]} {surv[1]} {surv[2]}">
      <geom name="survivor_geom" type="capsule" size="0.18 0.35"
            rgba="1.0 0.25 0.25 1" contype="0" conaffinity="0"/>
      <!-- glowing ring at feet -->
      <geom type="cylinder" size="0.5 0.02" pos="0 0 -0.55"
            rgba="1.0 0.9 0.0 0.7" contype="0" conaffinity="0"/>
    </body>

    <!-- Robot (sphere, 3 sliding joints = free 3-D motion) -->
    <body name="robot" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jz" type="slide" axis="0 0 1" range="0.2 6.0" damping="5"/>
      <geom name="robot_geom" type="sphere" size="0.3"
            rgba="0.15 0.5 1.0 1" mass="1" contype="1" conaffinity="1"/>
      <!-- direction indicator nub -->
      <geom type="sphere" size="0.1" pos="0.3 0 0"
            rgba="0.0 1.0 0.5 1" mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <!-- Velocity actuators give direct speed control — easier for RL -->
  <actuator>
    <velocity name="vx" joint="jx" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
    <velocity name="vy" joint="jy" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
    <velocity name="vz" joint="jz" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
  </actuator>
</mujoco>
"""
    return xml


# ── Environment ──────────────────────────────────────────────────────────────

class DisasterEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(self, scene: dict = None, render_mode: str = "rgb_array"):
        super().__init__()
        self.scene       = scene or DEFAULT_SCENE
        self.render_mode = render_mode
        self._step_count = 0
        self._prev_dist = None
        self._robot_start = np.array(_vec3(
            self.scene.get("robot_start", DEFAULT_SCENE["robot_start"])
        ), dtype=np.float64)

        # Cache survivor position (fixed during episode)
        self._surv_pos = np.array(_vec3(
            self.scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"])
        ), dtype=np.float64)
        self._hazards = [
            {
                "center": np.array(_vec3(h.get("center", [0, 0, 0]), 0.0))[:2],
                "radius": float(h.get("radius", 1.0)),
            }
            for h in self.scene.get("hazards", DEFAULT_SCENE["hazards"])
        ]

        # Build and load MuJoCo model
        self._xml = _build_xml(self.scene)
        self._model = mujoco.MjModel.from_xml_string(self._xml)
        self._data  = mujoco.MjData(self._model)

        # Robot joint / qpos indices (jx=0, jy=1, jz=2 relative to body)
        self._jx_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
        self._jy_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
        self._jz_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jz")

        # Spaces
        obs_high = np.array(
            [WORLD_HALF]*3 + [WORLD_HALF]*3 + [WORLD_HALF*2]*3 + [WORLD_HALF*2] + [5.0]*3,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # Renderer (lazy)
        self._renderer = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _robot_pos(self) -> np.ndarray:
        """Return robot xyz from joint qpos offsets (sliding joints)."""
        start = self._model.jnt_qposadr[self._jx_id]
        return self._data.qpos[start:start+3].copy()

    def _robot_vel(self) -> np.ndarray:
        start = self._model.jnt_dofadr[self._jx_id]
        return self._data.qvel[start:start+3].copy()

    def _get_obs(self) -> np.ndarray:
        rpos = self._robot_pos()
        rvel = self._robot_vel()
        rel = self._surv_pos - rpos
        dist = np.linalg.norm(rpos - self._surv_pos)
        return np.concatenate([rpos, self._surv_pos, rel, [dist], rvel]).astype(np.float32)

    def _in_hazard(self, rpos: np.ndarray) -> bool:
        rxy = rpos[:2]
        return any(
            np.linalg.norm(rxy - h["center"]) < h["radius"]
            for h in self._hazards
        )

    def _reconfigure(self, scene: dict):
        """Hot-swap scene — rebuild model without re-creating the env object."""
        self.scene    = scene
        self._robot_start = np.array(_vec3(scene.get("robot_start", DEFAULT_SCENE["robot_start"])), dtype=np.float64)
        self._surv_pos = np.array(_vec3(scene.get("survivor_pos", DEFAULT_SCENE["survivor_pos"])))
        self._hazards  = [
            {
                "center": np.array(_vec3(h.get("center", [0, 0, 0]), 0.0))[:2],
                "radius": float(h.get("radius", 1.0)),
            }
            for h in scene.get("hazards", DEFAULT_SCENE["hazards"])
        ]
        self._xml    = _build_xml(scene)
        self._model  = mujoco.MjModel.from_xml_string(self._xml)
        self._data   = mujoco.MjData(self._model)
        self._jx_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jx")
        self._jy_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jy")
        self._jz_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "jz")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        start = self._model.jnt_qposadr[self._jx_id]
        self._data.qpos[start:start+3] = self._robot_start
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._robot_pos() - self._surv_pos))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._data.ctrl[:] = action * ACTION_SPEED
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        rpos = self._robot_pos()
        dist = float(np.linalg.norm(rpos - self._surv_pos))

        # Reward shaping: direct progress dominates, with a small distance cost
        # so the policy learns to keep moving until the survivor is reached.
        progress = 0.0 if self._prev_dist is None else self._prev_dist - dist
        self._prev_dist = dist
        reward = progress * 10.0 - dist * 0.02
        if self._in_hazard(rpos):
            reward -= HAZARD_PEN
        # Out-of-bounds soft penalty
        oob = max(0.0, np.max(np.abs(rpos[:2])) - WORLD_HALF + 1.0)
        reward -= oob * 0.5

        reached   = dist < REACH_DIST
        truncated = self._step_count >= MAX_STEPS
        if reached:
            reward += REACH_BONUS

        info = {"reached": reached, "dist": dist, "steps": self._step_count}
        return self._get_obs(), reward, reached, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._renderer.update_scene(self._data, camera="track_robot")
            return self._renderer.render()

        elif self.render_mode == "human":
            mujoco.viewer.launch(self._model, self._data)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── Camera definition (add to XML via a second pass if needed) ───────────────
# We inject a tracking camera by overriding _build_xml's camera section below.

def _build_xml(scene: dict) -> str:  # noqa: F811  (redefine with camera)
    surv  = _vec3(scene.get("survivor_pos",  DEFAULT_SCENE["survivor_pos"]))
    obs   = scene.get("obstacles", DEFAULT_SCENE["obstacles"])
    haz   = scene.get("hazards",   DEFAULT_SCENE["hazards"])

    obs_xml = ""
    for i, o in enumerate(obs):
        p = _vec3(o["pos"], default_z=1.0)
        s = list(o.get("size", [0.5, 0.5, 1.0]))
        if len(s) == 2:
            s.append(1.0)
        s = [float(v) for v in s[:3]]
        obs_xml += (
            f'    <geom name="obs_{i}" type="box" '
            f'pos="{p[0]} {p[1]} {p[2]}" '
            f'size="{s[0]} {s[1]} {s[2]}" '
            f'rgba="0.55 0.27 0.07 1" contype="0" conaffinity="0"/>\n'
        )

    haz_xml = ""
    for i, h in enumerate(haz):
        c = _vec3(h.get("center", [0, 0, 0]), default_z=0.0)
        r = float(h.get("radius", 1.0))
        haz_xml += (
            f'    <geom name="haz_{i}" type="cylinder" '
            f'pos="{c[0]} {c[1]} 0.01" size="{r} 0.01" '
            f'rgba="1.0 0.15 0.15 0.4" contype="0" conaffinity="0"/>\n'
        )

    xml = f"""
<mujoco model="disaster_rescue">
  <option timestep="0.02" gravity="0 0 0" integrator="RK4"/>

  <asset>
    <texture name="checker" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.65 0.65 0.65" rgb2="0.45 0.45 0.45"/>
    <material name="grid_mat" texture="checker" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="{WORLD_HALF} {WORLD_HALF} 0.1"
          pos="0 0 0" material="grid_mat" contype="1" conaffinity="1"/>

    <light name="sky"   pos="0 0 15"  dir="0 0 -1"    diffuse="0.85 0.85 0.85" specular="0.2 0.2 0.2"/>
    <light name="front" pos="0 -12 8" dir="0 0.8 -0.6" diffuse="0.5 0.5 0.5"/>

{obs_xml}
{haz_xml}

    <!-- Survivor -->
    <body name="survivor" pos="{surv[0]} {surv[1]} {surv[2]}">
      <geom name="survivor_body" type="capsule" size="0.18 0.35"
            rgba="1.0 0.25 0.25 1" contype="0" conaffinity="0"/>
      <geom name="survivor_ring" type="cylinder" size="0.55 0.02" pos="0 0 -0.55"
            rgba="1.0 0.9 0.0 0.8" contype="0" conaffinity="0"/>
    </body>

    <!-- Humanoid robot with 3 sliding joints for 3D movement -->
    <body name="robot" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-{WORLD_HALF} {WORLD_HALF}" damping="5"/>
      <joint name="jz" type="slide" axis="0 0 1" range="0.2 6.0"                    damping="5"/>

      <!-- Torso -->
      <geom name="torso" type="capsule" size="0.18 0.4" pos="0 0 0.4"
            rgba="0.1 0.45 1.0 1" mass="2" contype="1" conaffinity="1"/>

      <!-- Head -->
      <geom name="head" type="sphere" size="0.15" pos="0 0 0.85"
            rgba="0.2 0.5 1.0 1" mass="0.5" contype="1" conaffinity="1"/>

      <!-- Left arm -->
      <geom name="l_arm" type="capsule" size="0.08 0.3" pos="-0.25 0 0.6"
            rgba="0.15 0.4 0.95 1" mass="0.3" contype="1" conaffinity="1"/>

      <!-- Right arm -->
      <geom name="r_arm" type="capsule" size="0.08 0.3" pos="0.25 0 0.6"
            rgba="0.15 0.4 0.95 1" mass="0.3" contype="1" conaffinity="1"/>

      <!-- Left leg -->
      <geom name="l_leg" type="capsule" size="0.1 0.35" pos="-0.12 0 0.05"
            rgba="0.25 0.35 0.9 1" mass="0.4" contype="1" conaffinity="1"/>

      <!-- Right leg -->
      <geom name="r_leg" type="capsule" size="0.1 0.35" pos="0.12 0 0.05"
            rgba="0.25 0.35 0.9 1" mass="0.4" contype="1" conaffinity="1"/>

      <!-- Direction indicator eye -->
      <geom name="eye" type="sphere" size="0.06" pos="0 0.15 0.85"
            rgba="0.0 1.0 0.4 1" mass="0" contype="0" conaffinity="0"/>

      <!-- Tracking camera mounted on robot -->
      <camera name="track_robot" pos="0 -6 4" xyaxes="1 0 0 0 0.4 1" mode="track"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="vx" joint="jx" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
    <velocity name="vy" joint="jy" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
    <velocity name="vz" joint="jz" kv="50" ctrllimited="true" ctrlrange="-{ACTION_SPEED} {ACTION_SPEED}"/>
  </actuator>
</mujoco>
"""
    return xml
