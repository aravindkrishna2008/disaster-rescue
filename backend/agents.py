from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-flash-lite-latest"
DEFAULT_BASE_AGENT = DEFAULT_MODEL
DEFAULT_SMOKE_TEST_MODEL = DEFAULT_MODEL
DEFAULT_THINKING_BUDGET = 256
ENV_PATHS = (
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).with_name(".env"),
)


class AgentResult:
    """Small result object matching the existing ScenarioAgent expectations."""

    def __init__(self, output_text: str, events: list[Any] | None = None) -> None:
        self.output_text = output_text
        self.events = events or []


def load_env_file(paths: tuple[Path, ...] = ENV_PATHS) -> None:
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    for path in paths:
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def configure_google_api_key() -> str:
    """Return a configured API key and expose it using ADK's GOOGLE_API_KEY name."""
    load_env_file()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "GEMINI_API_KEY":
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        return api_key

    raise RuntimeError(
        "Gemini API key is not configured. Set GOOGLE_API_KEY in your shell "
        "or create a .env file from .env.example."
    )


def create_low_thinking_config(response_mime_type: str | None = None) -> Any:
    """Create ADK model config for Gemini calls."""
    from google.genai import types

    if response_mime_type is None:
        return types.GenerateContentConfig()
    return types.GenerateContentConfig(response_mime_type=response_mime_type)


def create_low_thinking_planner() -> Any:
    """Create ADK's low-budget built-in Gemini planner."""
    from google.adk.planners import BuiltInPlanner
    from google.genai import types

    return BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_budget=DEFAULT_THINKING_BUDGET,
            include_thoughts=False,
        )
    )


@dataclass
class ManagedAgent:
    """Reusable embedded ADK agent wrapper."""

    agent_id: str
    system_prompt: str
    base_agent: str = DEFAULT_BASE_AGENT
    base_environment: dict[str, Any] | str = field(
        default_factory=lambda: {"type": "local"}
    )
    tools: list[Any] | None = None
    client: Any | None = None
    description: str | None = None
    model: str | None = None
    output_mime_type: str | None = None
    session_service: Any | None = None
    runner: Any | None = None
    _adk_agent: Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def with_inline_sources(
        cls,
        agent_id: str,
        system_prompt: str,
        sources: list[dict[str, Any]],
        *,
        base_agent: str = DEFAULT_BASE_AGENT,
        tools: list[Any] | None = None,
        client: Any | None = None,
        description: str | None = None,
    ) -> ManagedAgent:
        """Keep compatibility with older call sites that passed source metadata."""
        return cls(
            agent_id=agent_id,
            system_prompt=system_prompt,
            base_agent=base_agent,
            base_environment={"type": "local", "sources": sources},
            tools=tools,
            client=client,
            description=description,
        )

    def create(self) -> Any:
        """Create the embedded ADK agent and runner."""
        configure_google_api_key()
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        self._adk_agent = LlmAgent(
            name=self.agent_id.replace("-", "_"),
            model=self.model or self.base_agent or DEFAULT_MODEL,
            instruction=self.system_prompt,
            description=self.description or "",
            generate_content_config=create_low_thinking_config(
                response_mime_type=self.output_mime_type
            ),
            planner=create_low_thinking_planner(),
            tools=self.tools or [],
        )
        if self.session_service is None:
            self.session_service = InMemorySessionService()
        if self.runner is None:
            self.runner = Runner(
                agent=self._adk_agent,
                app_name=self.agent_id,
                session_service=self.session_service,
            )
        return self._adk_agent

    def run(
        self,
        prompt: str,
        *,
        environment: dict[str, Any] | str = "local",
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Invoke the embedded ADK agent and return text plus raw events."""
        if system_prompt is not None:
            original_prompt = self.system_prompt
            self.system_prompt = system_prompt
            self._adk_agent = None
            self.runner = None
            try:
                return self.run(prompt, environment=environment, tools=tools, **kwargs)
            finally:
                self.system_prompt = original_prompt
                self._adk_agent = None
                self.runner = None

        if tools is not None:
            self.tools = tools
            self._adk_agent = None
            self.runner = None

        if self.runner is None or self._adk_agent is None:
            self.create()

        _raise_if_running_loop()
        return _run_sync(self._run_async(prompt, **kwargs))

    def run_text(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the agent and return only the text output."""
        return self.run(prompt, **kwargs).output_text

    async def _run_async(self, prompt: str, **kwargs: Any) -> AgentResult:
        from google.genai import types

        if self.runner is None:
            raise RuntimeError("ManagedAgent runner was not initialized.")

        user_id = str(kwargs.pop("user_id", "default-user"))
        session_id = str(kwargs.pop("session_id", f"{self.agent_id}-session"))
        timeout = kwargs.pop("timeout", None)
        await _ensure_session(self.session_service, self.agent_id, user_id, session_id)

        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        events: list[Any] = []
        output_text = ""

        async def collect_events() -> None:
            nonlocal output_text
            run_async = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                **kwargs,
            )
            async for event in run_async:
                events.append(event)
                text = _event_text(event)
                if text:
                    output_text = text

        if timeout is None:
            await collect_events()
        else:
            await asyncio.wait_for(collect_events(), timeout=float(timeout))

        if not output_text:
            raise RuntimeError("ADK returned an empty response.")
        return AgentResult(output_text=output_text, events=events)


# Backwards-compatible name for the original module-level class.
Agent = ManagedAgent


async def _ensure_session(
    session_service: Any,
    app_name: str,
    user_id: str,
    session_id: str,
) -> None:
    if session_service is None:
        return

    getter = getattr(session_service, "get_session", None)
    if getter is not None:
        existing = getter(app_name=app_name, user_id=user_id, session_id=session_id)
        if inspectable_awaitable(existing):
            existing = await existing
        if existing is not None:
            return

    creator = getattr(session_service, "create_session", None)
    if creator is None:
        return
    created = creator(app_name=app_name, user_id=user_id, session_id=session_id)
    if inspectable_awaitable(created):
        await created


def inspectable_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def _event_text(event: Any) -> str:
    if hasattr(event, "is_final_response") and not event.is_final_response():
        return ""

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    text_parts = [getattr(part, "text", "") for part in parts]
    text = "".join(part for part in text_parts if part)
    return text or getattr(event, "output_text", "") or getattr(event, "text", "")


def _run_sync(coro: Any) -> Any:
    return asyncio.run(coro)


def _raise_if_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("ManagedAgent.run cannot be called from a running event loop.")


def verify_api_key(model: str = DEFAULT_SMOKE_TEST_MODEL) -> str:
    """Make a tiny ADK request to verify the configured API key works."""
    agent = ManagedAgent(
        agent_id="api-key-verification-agent",
        system_prompt="Reply exactly as requested.",
        base_agent=model,
    )
    return agent.run_text("Reply with only: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Google ADK Gemini setup.")
    parser.add_argument(
        "--model",
        default=DEFAULT_SMOKE_TEST_MODEL,
        help="Gemini model to use for the API-key smoke test.",
    )
    args = parser.parse_args()

    try:
        api_result = verify_api_key(model=args.model).strip()
    except Exception as exc:
        print(f"Google ADK Gemini verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Google ADK Gemini key works. Smoke-test response: {api_result}")


if __name__ == "__main__":
    main()
