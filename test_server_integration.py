"""
Issue #19: Test server independently with mocked Queue + fake robot thread.
Run: python test_server_integration.py
"""
import threading
import asyncio
from multiprocessing import Queue, Manager
from unittest import mock


def fake_robot(goal_queue, shared_result):
    """Simulates robot_process.py — picks up goal, marks done immediately."""
    while True:
        target_id = goal_queue.get()
        if target_id is None:
            break
        shared_result.update({"done": True, "reached": True, "steps": 42})


def run():
    manager = Manager()
    shared_result = manager.dict()
    goal_queue = Queue()

    # Start fake robot in background thread
    robot = threading.Thread(target=fake_robot, args=(goal_queue, shared_result), daemon=True)
    robot.start()

    from server import create_app, CommandRequest

    app = create_app(goal_queue, shared_result)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/command")

    with mock.patch(
        "gemini_client.get_gemini_target",
        return_value={"target_id": "child", "confidence": 0.9, "reason": "child first"},
    ):
        result = asyncio.run(route.endpoint(CommandRequest(text="save the child")))

    print("Response:", result)
    assert result["reached"] is True, f"expected reached=True, got {result}"
    assert result["steps"] == 42, f"expected steps=42, got {result}"
    assert result["target_id"] == "child"
    print("PASS — POST /command returns {reached: true, steps: 42}")

    # Cleanup
    goal_queue.put(None)
    manager.shutdown()


if __name__ == "__main__":
    run()
