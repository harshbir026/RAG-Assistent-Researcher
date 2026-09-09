# FastAPI async request handling

FastAPI runs on an ASGI server (usually Uvicorn). When a path operation is
declared `async def`, FastAPI awaits it on the event loop. When it is a
plain `def`, FastAPI runs it in a threadpool so blocking work does not freeze
other connections.

```python
import httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/upstream")
async def call_upstream():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://httpbin.org/get")
        return r.json()
```

Rules of thumb:

- Use `async def` + async libraries (`httpx`, `asyncpg`, SQLAlchemy 2.0
  async engine) for I/O-bound work. Many concurrent requests share one
  process because they yield at `await`.
- Use sync `def` for blocking libraries (`requests`, `psycopg2`,
  `time.sleep`). FastAPI will offload them; you still pay a thread.
- Never call blocking I/O inside `async def` without `asyncio.to_thread`.
  That stalls the loop for every other request on that worker.

Contrast with a typical synchronous Flask view: Flask+Werkzeug handles one
request per worker thread (or process). Concurrency comes from the number of
threads, not from cooperative awaiting. FastAPI's async path therefore
handles more waiting connections per process; Flask's sync path is simpler
when every library you use is blocking.
