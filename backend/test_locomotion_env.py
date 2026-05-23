"""Smoke tests for the G1 locomotion DisasterEnv."""

import numpy as np

from disaster_env import DisasterEnv


def assert_basic_spaces(env: DisasterEnv, obs: np.ndarray) -> None:
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (15,)
    assert np.isfinite(obs).all()


def step_zero_residual(env: DisasterEnv, steps: int) -> tuple[float, dict]:
    obs, _ = env.reset()
    assert_basic_spaces(env, obs)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    start_dist = None
    final_info = {}
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert "base_pos" in info
        assert len(info["base_pos"]) == 3
        assert info["z"] > 0.0
        if start_dist is None:
            start_dist = float(info["dist"])
        final_info = info
        if terminated or truncated:
            break

    assert start_dist is not None
    return start_dist, final_info


def run():
    env = DisasterEnv(render_mode="rgb_array", curriculum_stage="natural_target", assist_scale=0.05)
    obs, _ = env.reset()
    assert_basic_spaces(env, obs)
    assert len(env._path_waypoints) >= 2
    for geom_id in env._obstacle_geom_ids:
        assert int(env._model.geom_contype[geom_id]) == 1
        assert int(env._model.geom_conaffinity[geom_id]) == 1

    start_dist, info = step_zero_residual(env, steps=80)
    assert not info["fallen"]
    assert info["gait_phase"] > 0.0
    assert info["obstacle_contacts"] == 0
    assert info["hazard_steps"] == 0
    assert np.isfinite(info["mean_stance_slip"])
    assert np.isfinite(info["gait_score"])
    assert float(info["dist"]) <= start_dist + 0.25, (start_dist, info["dist"])
    assert info["curriculum_stage"] == "natural_target"
    assert info["assist_enabled"] is True
    assert info["assist_scale"] == 0.05
    assert info["balance_assist_scale"] == 0.05

    random_action = env.action_space.sample()
    obs, reward, *_ = env.step(random_action)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    env.close()

    stand_env = DisasterEnv(render_mode="rgb_array", curriculum_stage="stand", assist_enabled=True)
    _, stand_info = step_zero_residual(stand_env, steps=100)
    assert not stand_info["fallen"]
    assert stand_info["curriculum_stage"] == "stand"
    assert abs(stand_info["base_pos"][0] + 4.0) < 0.35
    assert abs(stand_info["base_pos"][1] + 4.0) < 0.35
    assert stand_info["assist_scale"] == 1.0
    assert stand_info["balance_assist_scale"] == 1.0
    stand_env.close()

    print("PASS - G1 locomotion environment smoke test")


if __name__ == "__main__":
    run()
