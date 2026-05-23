from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multiprocessing import Queue
import asyncio

SURVIVORS = {
    "child": (3.2, -1.5),
    "adult": (-2.1, 4.3),
}

_STATIC = Path(__file__).parent / "static"


class CommandRequest(BaseModel):
    text: str


def create_app(goal_queue: Queue, shared_result) -> FastAPI:
    app = FastAPI(title="GR-ER Ground Rescue")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text()

    @app.post("/command")
    async def command(req: CommandRequest):
        from gemini_client import get_gemini_target, SURVIVORS as _SURVIVORS

        result = get_gemini_target(req.text, _SURVIVORS)

        shared_result["done"] = False
        goal_queue.put(result["target_id"])

        # Poll without blocking the event loop — 30s max
        for _ in range(300):
            await asyncio.sleep(0.1)
            if shared_result.get("done"):
                break

        return {
            "target_id": result["target_id"],
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "reached": shared_result.get("reached", False),
            "steps": shared_result.get("steps", 0),
        }

    return app


# Dev mode: uvicorn server:app --reload
from multiprocessing import Manager as _Manager

_manager = _Manager()
app = create_app(Queue(), _manager.dict())
