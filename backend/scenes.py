"""Fixed disaster scenes used for PPO training and GIF generation."""

from __future__ import annotations

from copy import deepcopy


GENERATED_SCENES: list[dict] = [
    {
        "name": "earthquake_corridor",
        "robot_start": [-5.5, -5.0, 1.0],
        "survivor_pos": [5.0, 4.5, 1.0],
        "obstacles": [
            {"pos": [-2.5, -1.5, 1.0], "size": [0.35, 2.0, 1.0]},
            {"pos": [0.8, 1.2, 0.8], "size": [0.4, 1.6, 0.8]},
            {"pos": [3.0, -2.3, 1.1], "size": [0.5, 0.8, 1.1]},
        ],
        "hazards": [
            {"center": [-0.5, -3.0, 0.0], "radius": 0.9},
            {"center": [2.2, 2.7, 0.0], "radius": 1.0},
        ],
        "difficulty": "medium",
    },
    {
        "name": "flooded_hospital",
        "robot_start": [5.5, -5.2, 1.0],
        "survivor_pos": [-4.8, 4.8, 1.0],
        "obstacles": [
            {"pos": [2.2, -1.8, 1.0], "size": [1.2, 0.35, 1.0]},
            {"pos": [-0.7, 0.4, 1.0], "size": [0.4, 1.6, 1.0]},
            {"pos": [-3.1, 2.8, 0.8], "size": [0.7, 0.45, 0.8]},
        ],
        "hazards": [
            {"center": [2.6, 1.4, 0.0], "radius": 1.1},
            {"center": [-2.0, -2.8, 0.0], "radius": 1.0},
        ],
        "difficulty": "hard",
    },
    {
        "name": "wildfire_shelter",
        "robot_start": [-5.8, 4.8, 1.0],
        "survivor_pos": [5.2, -4.6, 1.0],
        "obstacles": [
            {"pos": [-2.8, 2.0, 1.0], "size": [0.6, 0.5, 1.0]},
            {"pos": [0.0, 0.0, 1.0], "size": [0.45, 2.2, 1.0]},
            {"pos": [2.6, -2.1, 1.0], "size": [1.0, 0.35, 1.0]},
        ],
        "hazards": [
            {"center": [-1.5, -1.6, 0.0], "radius": 1.0},
            {"center": [3.2, 1.7, 0.0], "radius": 0.85},
        ],
        "difficulty": "medium",
    },
    {
        "name": "collapsed_bridge",
        "robot_start": [5.2, 5.3, 1.0],
        "survivor_pos": [-5.0, -4.9, 1.0],
        "obstacles": [
            {"pos": [2.0, 2.0, 1.0], "size": [0.35, 1.6, 1.0]},
            {"pos": [-0.5, -0.2, 0.9], "size": [1.3, 0.35, 0.9]},
            {"pos": [-3.4, -2.8, 1.0], "size": [0.55, 1.0, 1.0]},
        ],
        "hazards": [
            {"center": [0.8, -2.5, 0.0], "radius": 1.0},
            {"center": [-2.2, 1.5, 0.0], "radius": 0.9},
        ],
        "difficulty": "hard",
    },
    {
        "name": "chemical_plant",
        "robot_start": [-6.0, 0.0, 1.0],
        "survivor_pos": [5.6, 0.8, 1.0],
        "obstacles": [
            {"pos": [-2.7, 0.5, 1.1], "size": [0.45, 1.4, 1.1]},
            {"pos": [0.0, -1.6, 1.0], "size": [1.1, 0.35, 1.0]},
            {"pos": [2.8, 1.6, 1.0], "size": [0.5, 1.2, 1.0]},
        ],
        "hazards": [
            {"center": [-0.6, 2.4, 0.0], "radius": 1.1},
            {"center": [2.0, -2.7, 0.0], "radius": 0.95},
        ],
        "difficulty": "medium",
    },
    {
        "name": "downtown_rubble",
        "robot_start": [0.0, -6.0, 1.0],
        "survivor_pos": [0.6, 5.6, 1.0],
        "obstacles": [
            {"pos": [-2.2, -2.0, 1.0], "size": [1.0, 0.35, 1.0]},
            {"pos": [1.6, 0.4, 1.0], "size": [0.4, 1.7, 1.0]},
            {"pos": [-1.6, 3.0, 0.9], "size": [0.7, 0.4, 0.9]},
        ],
        "hazards": [
            {"center": [2.8, -2.2, 0.0], "radius": 1.0},
            {"center": [-3.0, 1.5, 0.0], "radius": 0.9},
        ],
        "difficulty": "medium",
    },
]


def get_scene(index: int) -> dict:
    """Return a mutable copy of one fixed generated scene."""
    return deepcopy(GENERATED_SCENES[index % len(GENERATED_SCENES)])
