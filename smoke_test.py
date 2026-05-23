from __future__ import annotations

import argparse
from pathlib import Path

from scenario_agent import ScenarioAgent, SurvivorProfile

TEMP_DIR = Path("tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test ScenarioAgent.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompts, raw agent output, and repaired scene JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for the managed-agent interaction. Defaults to 45.",
    )
    args = parser.parse_args()

    TEMP_DIR.mkdir(exist_ok=True)

    agent = ScenarioAgent()
    agent.create()

    scene = agent.generate_scene(
        "Collapsed apartment after earthquake with no fire hazards.",
        survivor_count=2,
        survivor_profiles=[
            SurvivorProfile(name="Maya", type="baby", priority="critical"),
            SurvivorProfile(name="Luis", type="adult", priority="medium"),
        ],
        difficulty="easy",
        debug=args.debug,
        timeout=args.timeout,
    )

    json_path = scene.save_json(TEMP_DIR / "generated_scene.json")
    html_path = scene.save_preview_html(TEMP_DIR / "generated_scene.html")

    print(f"Wrote scene JSON: {json_path}")
    print(f"Wrote scene preview: {html_path}")


if __name__ == "__main__":
    main()
