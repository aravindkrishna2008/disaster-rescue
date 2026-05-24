from __future__ import annotations

import json

from agents import DEFAULT_MODEL, ManagedAgent, configure_google_api_key

SURVIVORS = {
    "child": (3.2, -1.5),
    "adult": (-2.1, 4.3),
}


def generate_scene_prompt(
    *,
    difficulty: str,
    survivor_count: int,
    theme: str | None,
) -> str:
    prompt_theme = (theme or "").strip()
    fallback_theme = prompt_theme or "urban collapse"
    fallback = (
        f"{fallback_theme}. Generate a {difficulty} disaster rescue scene with "
        f"{survivor_count} survivor{'s' if survivor_count != 1 else ''}. Include dense rubble, "
        "hazards, and a reachable robot start."
    )

    configure_google_api_key()
    agent = ManagedAgent(
        agent_id="rescue-scene-prompt-writer",
        system_prompt=_scene_prompt_system_prompt(),
        base_agent=DEFAULT_MODEL,
        description="Writes concise rescue-scene generation prompts.",
    )
    response_text = agent.run_text(
        _scene_prompt_user_prompt(
            difficulty=difficulty,
            survivor_count=survivor_count,
            theme=prompt_theme,
        )
    )
    cleaned = _clean_text_response(response_text)
    return cleaned or fallback


def get_gemini_target(command: str, survivors: dict) -> dict:
    """Return {"target_id": str, "confidence": float, "reason": str} via ADK."""
    try:
        configure_google_api_key()
    except Exception:
        return {
            "target_id": _fallback_target(survivors),
            "confidence": 0.5,
            "reason": "stub (no GOOGLE_API_KEY)",
        }

    try:
        agent = ManagedAgent(
            agent_id="rescue-target-selector",
            system_prompt=_target_system_prompt(survivors),
            base_agent=DEFAULT_MODEL,
            output_mime_type="application/json",
            description="Selects the next survivor target for the rescue robot.",
        )
        response_text = agent.run_text(_target_user_prompt(command))
        return _parse_target_response(response_text, survivors)
    except Exception as exc:
        print(f"[gemini_client] fallback triggered: {exc}")
        return {
            "target_id": _fallback_target(survivors),
            "confidence": 0.0,
            "reason": f"fallback: {exc}",
        }


def _target_system_prompt(survivors: dict) -> str:
    survivor_desc = "\n".join(
        f"- {survivor_id}: position {position}"
        for survivor_id, position in survivors.items()
    )
    valid_targets = " | ".join(f'"{survivor_id}"' for survivor_id in survivors)
    return f"""You are controlling a disaster rescue robot.
Survivors in the environment:
{survivor_desc}

Choose which survivor to rescue next.
Respond only with JSON matching this schema:
{{"target_id": {valid_targets}, "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def _target_user_prompt(command: str) -> str:
    return f'Natural language command: "{command}"'


def _scene_prompt_system_prompt() -> str:
    return """You write one concise natural-language prompt for a rescue-scene generator.
Return plain text only.
Keep it to one sentence.
Describe the environment, collapse pattern, hazards, survivor situation, and traversal constraints.
If the user supplied a theme hint, use it as the primary setting and visual motif."""


def _scene_prompt_user_prompt(
    *,
    difficulty: str,
    survivor_count: int,
    theme: str,
) -> str:
    theme_text = theme if theme else "No theme hint was supplied."
    return (
        f"Difficulty: {difficulty}\n"
        f"Survivor count: {survivor_count}\n"
        f"Theme hint: {theme_text}\n"
        "Write the prompt now."
    )


def _parse_target_response(response_text: str, survivors: dict) -> dict:
    data = json.loads(_extract_json(response_text))
    target_id = data.get("target_id")
    if target_id not in survivors:
        raise ValueError(f"Unknown target_id: {target_id}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "target_id": target_id,
        "confidence": confidence,
        "reason": reason,
    }


def _extract_json(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Gemini response did not contain a JSON object.")
    return stripped[start : end + 1]


def _clean_text_response(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return " ".join(stripped.split())


def _fallback_target(survivors: dict) -> str:
    return next(iter(survivors), "child")
