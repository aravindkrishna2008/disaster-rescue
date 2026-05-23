"""
FastAPI backend for the Battle Angel console.

POST /command  body: {"text": "<operator order>"}
returns:       {target_id, confidence, reason, reached, steps, total_reward, gif_url}

Pipeline:
  1. Gemini picks which survivor ("child" | "adult") to rescue from the order
  2. Build a scene with that survivor's coords and run one PPO episode
  3. Return aggregate stats; the rendered GIF is exposed at /episode.gif

Episode rollouts are CPU-bound (PPO inference + MuJoCo), so we run them on a
thread via asyncio.to_thread to keep the event loop responsive.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from disaster_env import DEFAULT_SCENE
from gemini_client import SURVIVORS, get_gemini_target
from rescue_runner import run_episode

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_OUTPUT = _HERE / "output"
_OUTPUT.mkdir(exist_ok=True)

MODEL_PATH = os.environ.get(
    "BATTLE_ANGEL_MODEL", str(_HERE / "models" / "ppo_fixed_six_final")
)
MAX_STEPS = int(os.environ.get("BATTLE_ANGEL_MAX_STEPS", "300"))


class CommandRequest(BaseModel):
    text: str


def _scene_for_target(target_id: str) -> dict:
    """Clone DEFAULT_SCENE but plant the survivor at the chosen target coords."""
    if target_id not in SURVIVORS:
        raise ValueError(f"unknown target_id: {target_id!r}")
    x, y = SURVIVORS[target_id]
    return {
        **DEFAULT_SCENE,
        "survivor_pos": [float(x), float(y), 0.0],
    }


def _run_episode_for(target_id: str) -> dict:
    scene = _scene_for_target(target_id)
    gif_path = _OUTPUT / f"episode_{target_id}.gif"
    return run_episode(
        scene=scene,
        model_path=MODEL_PATH,
        gif_path=str(gif_path),
        max_steps=MAX_STEPS,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Battle Angel · Ground Rescue")

    # Allow direct calls from the Next.js dev server (the rewrite in
    # next.config.mjs hides this in prod, but CORS makes ad-hoc testing easy).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return (_STATIC / "index.html").read_text()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": MODEL_PATH, "max_steps": MAX_STEPS}

    @app.get("/episode.gif")
    def latest_gif():
        gifs = sorted(
            _OUTPUT.glob("episode_*.gif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not gifs:
            raise HTTPException(status_code=404, detail="no episode gif yet")
        return FileResponse(gifs[0], media_type="image/gif")

    @app.post("/command")
    async def command(req: CommandRequest) -> dict:
        decision = get_gemini_target(req.text, SURVIVORS)
        target_id = decision["target_id"]

        try:
            result = await asyncio.to_thread(_run_episode_for, target_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"model not found: {e}") from e

        return {
            "target_id": target_id,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason": decision.get("reason", ""),
            "reached": bool(result.get("reached", False)),
            "steps": int(result.get("steps", 0)),
            "total_reward": float(result.get("total_reward", 0.0)),
            "gif_url": "/episode.gif",
            "trajectory": result.get("trajectory", []),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
