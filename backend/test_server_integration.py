"""
Server integration smoke test with mocked Gemini and PPO rollout.
Run: uv run python test_server_integration.py
"""

import asyncio
from unittest import mock


def run():
    from server import CommandRequest, create_app

    app = create_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/command")

    with (
        mock.patch(
            "server.get_gemini_target",
            return_value={"target_id": "child", "confidence": 0.9, "reason": "child first"},
        ),
        mock.patch(
            "server._run_episode_for",
            return_value={
                "reached": True,
                "fallen": False,
                "steps": 42,
                "total_reward": 12.5,
                "final_dist": 0.4,
                "min_dist": 0.4,
                "obstacle_contacts": 0,
                "hazard_steps": 0,
                "mean_stance_slip": 0.12,
                "mean_assist_force": 4.5,
                "gait_score": 0.91,
                "assist_scale": 0.95,
                "balance_assist_scale": 0.85,
                "trajectory": [[0.0, 0.0, 0.79], [1.0, 1.0, 0.8]],
            },
        ),
    ):
        result = asyncio.run(route.endpoint(CommandRequest(text="save the child")))

    print("Response:", result)
    assert result["target_id"] == "child"
    assert result["confidence"] == 0.9
    assert result["reached"] is True
    assert result["fallen"] is False
    assert result["steps"] == 42
    assert result["total_reward"] == 12.5
    assert result["final_dist"] == 0.4
    assert result["min_dist"] == 0.4
    assert result["obstacle_contacts"] == 0
    assert result["hazard_steps"] == 0
    assert result["mean_stance_slip"] == 0.12
    assert result["mean_assist_force"] == 4.5
    assert result["gait_score"] == 0.91
    assert result["assist_scale"] == 0.95
    assert result["balance_assist_scale"] == 0.85
    assert result["gif_url"] == "/episode.gif"
    assert result["trajectory"][-1] == [1.0, 1.0, 0.8]
    print("PASS - POST /command returns mocked Gemini + rollout response")


if __name__ == "__main__":
    run()
