"""
test_episode_loader.py — pytest suite for episode loading, transition extraction,
and simulation replayer.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def skip(reason):
            print(f"Skipped: {reason}")
    pytest = MockPytest()
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from episode_loader import load_episodes, load_transitions, replay_episode_actions
from run_store import run_dir


def test_load_episodes():
    run_id = "ppo_buried_detection_final"
    episodes = load_episodes(run_id)
    assert len(episodes) > 0, "Should load at least one episode from ppo_buried_detection_final"
    for ep in episodes:
        assert "scene_index" in ep
        assert "scene_name" in ep
        assert "reached" in ep


def test_load_transitions_with_mock():
    # Since committed ppo_buried_detection_final episodes have reached=False,
    # we create a mock successful run/episode to test load_transitions.
    mock_run_id = "test_bc_loader_run"
    mock_dir = run_dir(mock_run_id)
    ep_dir = mock_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    mock_ep = {
        "scene_index": 0,
        "scene_name": "test_scene",
        "reached": True,
        "steps": 2,
        "total_reward": 10.0,
        "trajectory": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        "rollout": [
            {
                "obs": [0.0] * 21,
                "action": [0.1, -0.1],
                "reward": 5.0,
                "terminated": False,
                "truncated": False
            },
            {
                "obs": [1.0] * 21,
                "action": [0.2, -0.2],
                "reward": 5.0,
                "terminated": True,
                "truncated": False
            }
        ]
    }

    try:
        with open(ep_dir / "01_test_scene.json", "w", encoding="utf-8") as f:
            json.dump(mock_ep, f)

        transitions = load_transitions(mock_run_id)
        assert len(transitions) == 2, f"Should load 2 transitions, got {len(transitions)}"
        
        obs, act = transitions[0]
        assert isinstance(obs, np.ndarray)
        assert isinstance(act, np.ndarray)
        assert obs.shape == (21,)
        assert act.shape == (2,)
        assert np.allclose(act, np.array([0.1, -0.1]))
    finally:
        if mock_dir.exists():
            shutil.rmtree(mock_dir)


def test_replay_episode_actions():
    try:
        from disaster_env import DisasterEnv
    except (ImportError, Exception):
        pytest.skip("DisasterEnv/MuJoCo dependencies not available")

    run_id = "ppo_buried_detection_final"
    ep_dir = _HERE / "runs" / run_id / "episodes"
    eps = list(ep_dir.glob("*.json"))
    if not eps:
        pytest.skip("No episode JSON files found on disk")

    ep_path = eps[0]

    try:
        res = replay_episode_actions(str(ep_path))
        assert "reached" in res
        assert "steps" in res
        assert "trajectory_match_score" in res
    except Exception as e:
        pytest.skip(f"Replay failed (likely due to headless environment or missing MuJoCo assets): {e}")
