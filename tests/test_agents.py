from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import agents
from agents import DEFAULT_MODEL, DEFAULT_SMOKE_TEST_MODEL, Agent, ManagedAgent


class FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeContent:
    def __init__(self, text: str) -> None:
        self.parts = [FakePart(text)]


class FakeEvent:
    def __init__(self, text: str, final: bool = True) -> None:
        self.content = FakeContent(text)
        self._final = final

    def is_final_response(self) -> bool:
        return self._final


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_async(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield FakeEvent("ignored", final=False)
        yield FakeEvent("managed-agent-ok")


class FakeSessionService:
    def __init__(self) -> None:
        self.created: list[dict[str, str]] = []
        self.sessions: set[tuple[str, str, str]] = set()

    def get_session(self, app_name: str, user_id: str, session_id: str) -> object | None:
        key = (app_name, user_id, session_id)
        return object() if key in self.sessions else None

    def create_session(self, app_name: str, user_id: str, session_id: str) -> object:
        self.created.append(
            {"app_name": app_name, "user_id": user_id, "session_id": session_id}
        )
        self.sessions.add((app_name, user_id, session_id))
        return object()


def test_agent_alias_points_to_managed_agent() -> None:
    assert Agent is ManagedAgent


def test_managed_agent_defaults_to_flash_lite_model() -> None:
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
    )

    assert agent.base_agent == DEFAULT_MODEL
    assert DEFAULT_SMOKE_TEST_MODEL == DEFAULT_MODEL


def test_managed_agent_run_uses_adk_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agents, "configure_google_api_key", lambda: "key")
    fake_runner = FakeRunner()
    fake_sessions = FakeSessionService()
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        runner=fake_runner,
        session_service=fake_sessions,
    )
    agent._adk_agent = object()

    result = agent.run_text("Say ok", user_id="u1", session_id="s1")

    assert result == "managed-agent-ok"
    assert fake_sessions.created == [
        {"app_name": "verification-agent", "user_id": "u1", "session_id": "s1"}
    ]
    assert fake_runner.calls[0]["user_id"] == "u1"
    assert fake_runner.calls[0]["session_id"] == "s1"
    assert fake_runner.calls[0]["new_message"].parts[0].text == "Say ok"


def test_managed_agent_reuses_existing_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agents, "configure_google_api_key", lambda: "key")
    fake_runner = FakeRunner()
    fake_sessions = FakeSessionService()
    fake_sessions.sessions.add(("verification-agent", "u1", "s1"))
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        runner=fake_runner,
        session_service=fake_sessions,
    )
    agent._adk_agent = object()

    assert agent.run_text("Say ok", user_id="u1", session_id="s1") == "managed-agent-ok"
    assert fake_sessions.created == []


def test_managed_agent_raises_from_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agents, "configure_google_api_key", lambda: "key")
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        runner=FakeRunner(),
        session_service=FakeSessionService(),
    )
    agent._adk_agent = object()

    async def call_run_text() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            agent.run_text("Say ok")

    import asyncio

    asyncio.run(call_run_text())
