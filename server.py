from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from multiprocessing import Queue
import asyncio

SURVIVORS = {
    "child": (3.2, -1.5),
    "adult": (-2.1, 4.3),
}

HTML_UI = """<!DOCTYPE html>
<html>
<head>
  <title>Disaster Rescue Robot</title>
  <style>
    body { font-family: monospace; max-width: 600px; margin: 60px auto; padding: 0 20px; }
    h1 { font-size: 1.2rem; }
    input { width: 100%; padding: 8px; font-size: 1rem; box-sizing: border-box; }
    button { margin-top: 8px; padding: 8px 20px; font-size: 1rem; cursor: pointer; }
    pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; margin-top: 16px; }
  </style>
</head>
<body>
  <h1>Disaster Rescue Robot</h1>
  <input id="cmd" type="text" placeholder='e.g. "save the child first"' />
  <br/>
  <button onclick="send()">Send Command</button>
  <pre id="out">Response will appear here...</pre>
  <script>
    async function send() {
      const text = document.getElementById('cmd').value;
      document.getElementById('out').textContent = 'Sending...';
      const res = await fetch('/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      const data = await res.json();
      document.getElementById('out').textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>"""


class CommandRequest(BaseModel):
    text: str


def create_app(goal_queue: Queue, shared_result) -> FastAPI:
    app = FastAPI(title="Disaster Rescue Robot")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_UI

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
