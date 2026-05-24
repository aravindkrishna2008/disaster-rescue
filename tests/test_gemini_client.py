from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import gemini_client


class FakeManagedAgent:
    response_text = '{"target_id": "adult", "confidence": 0.82, "reason": "closest"}'

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_text(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response_text


def test_get_gemini_target_uses_adk_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_client, "configure_google_api_key", lambda: "key")
    monkeypatch.setattr(gemini_client, "ManagedAgent", FakeManagedAgent)

    result = gemini_client.get_gemini_target(
        "rescue the adult", {"child": (0, 0), "adult": (1, 1)}
    )

    assert result == {
        "target_id": "adult",
        "confidence": 0.82,
        "reason": "closest",
    }


def test_generate_scene_prompt_uses_adk_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_client, "configure_google_api_key", lambda: "key")
    class PromptAgent(FakeManagedAgent):
        response_text = "Collapsed chemical plant with leaking gas pockets and one trapped worker behind dense rubble."

    fake_agent = PromptAgent
    monkeypatch.setattr(gemini_client, "ManagedAgent", fake_agent)

    result = gemini_client.generate_scene_prompt(
        difficulty="hard",
        survivor_count=1,
        theme="chemical plant",
    )

    assert result == fake_agent.response_text


def test_generate_scene_prompt_strips_markdown_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini_client, "configure_google_api_key", lambda: "key")
    class PromptAgent(FakeManagedAgent):
        response_text = "```text\nFlooded parking garage with one survivor on a tilted slab.\n```"

    fake_agent = PromptAgent
    monkeypatch.setattr(gemini_client, "ManagedAgent", fake_agent)

    result = gemini_client.generate_scene_prompt(
        difficulty="medium",
        survivor_count=1,
        theme="flood",
    )

    assert result == "Flooded parking garage with one survivor on a tilted slab."


def test_get_gemini_target_falls_back_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_key() -> str:
        raise RuntimeError("missing")

    monkeypatch.setattr(gemini_client, "configure_google_api_key", raise_missing_key)

    result = gemini_client.get_gemini_target("save someone", {"adult": (1, 1)})

    assert result["target_id"] == "adult"
    assert result["confidence"] == 0.5
    assert result["reason"] == "stub (no GOOGLE_API_KEY)"


def test_parse_target_response_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target_id"):
        gemini_client._parse_target_response(
            '{"target_id": "dog", "confidence": 1, "reason": "bad"}',
            {"child": (0, 0)},
        )


def test_parse_target_response_handles_markdown_and_clamps_confidence() -> None:
    result = gemini_client._parse_target_response(
        '```json\n{"target_id": "child", "confidence": 8, "reason": 7}\n```',
        {"child": (0, 0)},
    )

    assert result == {"target_id": "child", "confidence": 1.0, "reason": "7"}
