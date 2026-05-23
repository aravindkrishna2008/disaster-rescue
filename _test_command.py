"""Smoke test for POST /command async route with mocked deps."""
import asyncio
from multiprocessing import Queue, Manager
from unittest import mock


def run():
    manager = Manager()
    shared = manager.dict()
    q = Queue()

    shared["done"] = True
    shared["reached"] = True
    shared["steps"] = 42

    from server import create_app, CommandRequest

    app = create_app(q, shared)

    route = next(r for r in app.routes if getattr(r, "path", None) == "/command")
    print("POST /command route registered:", route.path)

    with mock.patch(
        "gemini_client.get_gemini_target",
        return_value={"target_id": "child", "confidence": 0.95, "reason": "child needs help"},
    ):
        result = asyncio.run(route.endpoint(CommandRequest(text="save the child")))

    print("Response:", result)
    assert result["target_id"] == "child", result
    assert result["reached"] is True, result
    assert result["steps"] == 42, result
    print("All assertions passed.")
    manager.shutdown()


if __name__ == "__main__":
    run()
