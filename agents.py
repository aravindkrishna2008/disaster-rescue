from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai

DEFAULT_BASE_AGENT = "antigravity-preview-05-2026"
DEFAULT_SMOKE_TEST_MODEL = "gemini-2.5-flash"
ENV_PATH = Path(__file__).with_name(".env")


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Return a configured Gemini API key or raise a clear setup error."""
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "GEMINI_API_KEY":
        return api_key

    raise RuntimeError(
        "Gemini API key is not configured. Set GEMINI_API_KEY in your shell "
        "or create a .env file from .env.example."
    )


def create_genai_client(api_key: str | None = None) -> genai.Client:
    """Create a GenAI client using an explicit or environment-loaded API key."""
    return genai.Client(api_key=api_key or get_api_key())


@dataclass
class ManagedAgent:
    """Reusable wrapper for creating and invoking Gemini managed agents."""

    agent_id: str
    system_prompt: str
    base_agent: str = DEFAULT_BASE_AGENT
    base_environment: dict[str, Any] | str = field(
        default_factory=lambda: {"type": "remote"}
    )
    tools: list[dict[str, Any]] | None = None
    client: Any | None = None
    description: str | None = None
    _created_agent: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = create_genai_client()

    @property
    def _client(self) -> Any:
        if self.client is None:
            raise RuntimeError("ManagedAgent client was not initialized.")
        return self.client

    @classmethod
    def with_inline_sources(
        cls,
        agent_id: str,
        system_prompt: str,
        sources: list[dict[str, Any]],
        *,
        base_agent: str = DEFAULT_BASE_AGENT,
        tools: list[dict[str, Any]] | None = None,
        client: Any | None = None,
        description: str | None = None,
    ) -> ManagedAgent:
        """Create an agent configured with inline/repository/GCS sources."""
        return cls(
            agent_id=agent_id,
            system_prompt=system_prompt,
            base_agent=base_agent,
            base_environment={
                "type": "remote",
                "sources": sources,
            },
            tools=tools,
            client=client,
            description=description,
        )

    def create(self) -> Any:
        """Register this configuration as a Gemini managed agent."""
        create_kwargs: dict[str, Any] = {
            "id": self.agent_id,
            "base_agent": self.base_agent,
            "system_instruction": self.system_prompt,
            "base_environment": self.base_environment,
        }
        if self.description is not None:
            create_kwargs["description"] = self.description

        try:
            self._created_agent = self._client.agents.create(**create_kwargs)
        except Exception as exc:
            if not is_agent_conflict_error(exc):
                raise
            self._created_agent = self._client.agents.get(self.agent_id)
        return self._created_agent

    def run(
        self,
        prompt: str,
        *,
        environment: dict[str, Any] | str = "remote",
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the managed agent by ID and return the interaction result."""
        interaction_kwargs: dict[str, Any] = {
            "agent": self.agent_id,
            "input": prompt,
            "environment": environment,
            **kwargs,
        }
        if system_prompt is not None:
            interaction_kwargs["system_instruction"] = system_prompt
        interaction_tools = tools if tools is not None else self.tools
        if interaction_tools is not None:
            interaction_kwargs["tools"] = interaction_tools

        return self._client.interactions.create(**interaction_kwargs)

    def run_text(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the agent and return only the text output."""
        result = self.run(prompt, **kwargs)
        return result.output_text


# Backwards-compatible name for the original module-level class.
Agent = ManagedAgent


def is_agent_conflict_error(exc: Exception) -> bool:
    """Return True when the managed agent already exists remotely."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 409:
        return True

    message = str(exc).lower()
    return "409" in message and "already exists" in message


def verify_api_key(model: str = DEFAULT_SMOKE_TEST_MODEL) -> str:
    """Make a tiny Gemini request to verify the configured API key works."""
    client = create_genai_client()
    response = client.models.generate_content(
        model=model,
        contents="Reply with only: ok",
    )
    text = response.text or getattr(response, "output_text", "")
    if not text:
        raise RuntimeError("Gemini returned an empty smoke-test response.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Gemini agent setup.")
    parser.add_argument(
        "--model",
        default=DEFAULT_SMOKE_TEST_MODEL,
        help="Gemini model to use for the API-key smoke test.",
    )
    args = parser.parse_args()

    try:
        api_result = verify_api_key(model=args.model).strip()
    except Exception as exc:
        print(f"Gemini API key verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Gemini API key works. Smoke-test response: {api_result}")


if __name__ == "__main__":
    main()
