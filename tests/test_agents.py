from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import DEFAULT_BASE_AGENT, DEFAULT_SMOKE_TEST_MODEL, Agent, ManagedAgent


class FakeAgentsClient:
    def __init__(self) -> None:
        self.created_kwargs: dict[str, Any] | None = None
        self.fetched_id: str | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.created_kwargs = kwargs
        return {"id": kwargs["id"], "created": True}

    def get(self, agent_id: str) -> dict[str, Any]:
        self.fetched_id = agent_id
        return {"id": agent_id, "created": False}


class ConflictAgentsClient(FakeAgentsClient):
    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.created_kwargs = kwargs
        raise RuntimeError("409 Requested entity already exists")


class FakeInteractionsClient:
    def __init__(self) -> None:
        self.created_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.created_kwargs = kwargs

        class Result:
            output_text = "managed-agent-ok"

        return Result()


class FakeGenAIClient:
    def __init__(self) -> None:
        self.agents = FakeAgentsClient()
        self.interactions = FakeInteractionsClient()


class ConflictGenAIClient(FakeGenAIClient):
    def __init__(self) -> None:
        self.agents = ConflictAgentsClient()
        self.interactions = FakeInteractionsClient()


def test_agent_alias_points_to_managed_agent() -> None:
    assert Agent is ManagedAgent


def test_managed_agent_create_builds_expected_request() -> None:
    fake_client = FakeGenAIClient()
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        client=fake_client,
        description="Verification agent",
    )

    created = agent.create()

    assert created == {"id": "verification-agent", "created": True}
    assert fake_client.agents.created_kwargs == {
        "id": "verification-agent",
        "base_agent": DEFAULT_BASE_AGENT,
        "system_instruction": "You verify wiring.",
        "base_environment": {"type": "remote"},
        "description": "Verification agent",
    }


def test_managed_agent_create_fetches_existing_agent_on_conflict() -> None:
    fake_client = ConflictGenAIClient()
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        client=fake_client,
    )

    created = agent.create()

    assert created == {"id": "verification-agent", "created": False}
    assert fake_client.agents.fetched_id == "verification-agent"


def test_managed_agent_run_builds_expected_interaction_request() -> None:
    fake_client = FakeGenAIClient()
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        client=fake_client,
    )

    result = agent.run_text("Say ok", environment={"type": "remote"})

    assert result == "managed-agent-ok"
    assert fake_client.interactions.created_kwargs == {
        "agent": "verification-agent",
        "input": "Say ok",
        "environment": {"type": "remote"},
    }


def test_managed_agent_raises_when_client_is_missing() -> None:
    agent = ManagedAgent(
        agent_id="verification-agent",
        system_prompt="You verify wiring.",
        client=FakeGenAIClient(),
    )
    agent.client = None

    with pytest.raises(RuntimeError, match="client was not initialized"):
        agent.run_text("Say ok")


def test_managed_agent_loads_real_api_key_and_client_works() -> None:
    import os
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")

    agent = ManagedAgent(
        agent_id="real-client-verification-agent",
        system_prompt="You verify real client wiring.",
    )

    response = agent._client.models.generate_content(
        model=DEFAULT_SMOKE_TEST_MODEL,
        contents="Reply with only: ok",
    )

    assert response.text is not None
    assert response.text.strip().lower() == "ok"
