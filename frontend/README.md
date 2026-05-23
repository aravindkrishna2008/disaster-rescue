# GR-ER Frontend (Next.js)

Next.js port of `static/index.html`. The UI talks to the FastAPI backend (`server.py`) via `POST /command`.

## Dev

```bash
cd frontend
npm install
# point at a running FastAPI backend (defaults to http://127.0.0.1:8000)
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:3000. `/command` is proxied to `${BACKEND_URL}/command` via `next.config.mjs` rewrites — no CORS needed.

## Build

```bash
npm run build
npm start
```
