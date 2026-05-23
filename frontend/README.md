# Battle Angel Frontend (Next.js)

The UI talks to the FastAPI backend in `../backend/` via `POST /command`.

## Dev

```bash
# 1. start the backend (in another terminal, from the repo root)
cd ../backend
uv sync
uv run uvicorn server:app --host 127.0.0.1 --port 8000

# 2. start the frontend
cd frontend
npm install
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:3000. `/command` is proxied to `${BACKEND_URL}/command` via `next.config.mjs` rewrites — no CORS needed.

## Build

```bash
npm run build
npm start
```
