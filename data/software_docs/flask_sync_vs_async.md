# Flask synchronous request handling vs FastAPI async

Flask's default (Werkzeug) is WSGI and synchronous. A view function
`def index():` occupies a worker until it returns. Concurrency equals
worker count (processes × threads). Flask 2+ can define `async def` views,
but many extensions and `requests` remain blocking; an `async def` that
calls `requests.get` still blocks the loop.

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/items/<int:item_id>")
def read_item(item_id):
    # blocking: this thread is busy until the function returns
    return jsonify({"item_id": item_id})
```

FastAPI equivalent with non-blocking I/O:

```python
import httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    async with httpx.AsyncClient() as client:
        await client.get("https://example.com/ping")
    return {"item_id": item_id}
```

Comparison:

| | Flask (typical sync) | FastAPI (async handler) |
|---|---|---|
| Server model | WSGI, one request per thread | ASGI, many waiting requests per event loop |
| Path params | Converter in the URL rule (`<int:id>`) | Type hints (`item_id: int`) + validation |
| Blocking libraries | Natural fit | Must use `def` or `to_thread` |
| OpenAPI | Add-on (apispec, flasgger) | Generated from the same signatures |
| Best when | Simple apps, sync ORM, few connections | I/O-heavy APIs, many concurrent waits |

FastAPI does not magically make CPU work faster. It makes waiting cheaper.
If both apps call a blocking ORM on every request, threadpools in FastAPI
and threads in Flask look similar; the async advantage appears when handlers
await network or disk.
