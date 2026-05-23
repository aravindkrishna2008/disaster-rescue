from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path

from scenario_agent import SceneSpec


TEMP_DIR = Path("tmp")
DEFAULT_OUTPUT = TEMP_DIR / "scenario_preview.html"


def load_scene(path: str | Path) -> SceneSpec:
    scene_path = Path(path)
    return SceneSpec.model_validate_json(scene_path.read_text(encoding="utf-8"))


def write_preview(scene_path: str | Path, output_path: str | Path) -> Path:
    scene = load_scene(scene_path)
    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    scene.save_preview_html(preview_path)
    return preview_path


def serve_directory(directory: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving {directory} at http://127.0.0.1:{port}/")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a standalone 3D-ish HTML preview from a ScenarioAgent JSON scene."
    )
    parser.add_argument(
        "scene_json",
        help="Path to a JSON file produced by SceneSpec.save_json().",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"HTML preview path to write. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the output directory locally after writing the preview.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Local port to use with --serve. Defaults to 8000.",
    )
    args = parser.parse_args()

    preview_path = write_preview(args.scene_json, args.out).resolve()
    print(f"Wrote preview: {preview_path}")

    if args.serve:
        print(f"Open: http://127.0.0.1:{args.port}/{preview_path.name}")
        serve_directory(preview_path.parent, args.port)


if __name__ == "__main__":
    main()
