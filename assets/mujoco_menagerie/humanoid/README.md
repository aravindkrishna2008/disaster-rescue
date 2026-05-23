# Humanoid Target Asset

A detailed humanoid model for use as a target/survivor in the disaster rescue environment.

## Structure

The humanoid model includes:
- **Torso**: Central body with 3 DOF (rotations around x, y, z axes)
- **Head**: Head segment with neck joint for rotation
- **Arms**: Left and right arms with shoulder (2 DOF) and elbow (1 DOF) joints
- **Legs**: Left and right legs with hip (3 DOF), knee (1 DOF), and ankle (1 DOF) joints

## Body Parts

| Part | DOF | Mass | Purpose |
|------|-----|------|---------|
| Torso | 3 | 8.0 kg | Main body segment |
| Head | 1 | 1.5 kg | Head with neck rotation |
| Each Arm | 3 | 1.8 kg | Arms for interaction |
| Each Leg | 5 | 3.9 kg | Legs for support |

## Usage

To use this model as a target in the disaster rescue environment:

```python
# In disaster_env.py, replace the simple survivor body with:
<body name="survivor" pos="{survivor[0]} {survivor[1]} 0.8">
  <include file="assets/mujoco_menagerie/humanoid/humanoid.xml"/>
</body>
```

Or load it directly in Python:
```python
import mujoco

# Load humanoid model
humanoid_model = mujoco.MjModel.from_xml_file('assets/mujoco_menagerie/humanoid/humanoid.xml')
```

## Features

- Realistic human-like proportions
- Full articulated skeleton with 17 joints
- Capsule and sphere geometries for efficient collision detection
- Suitable for both visualization and physics simulation
- Compatible with grounded (2D) environments
