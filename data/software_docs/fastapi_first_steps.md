# FastAPI first steps

FastAPI is an ASGI web framework for building HTTP APIs in Python. A minimal
app constructs a `FastAPI()` instance and registers path operations with
decorators. The framework inspects function signatures to build an OpenAPI
schema automatically.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
```

This creates a basic app with one route. `GET /` returns JSON. Because the
handler is a plain function (not `async def`), FastAPI runs it in a threadpool
so it does not block the event loop. Prefer `async def` when the handler
awaits I/O; prefer sync `def` when it calls blocking libraries such as
classic SQLAlchemy or `requests`.

Run locally with Uvicorn: `uvicorn main:app --reload`. Interactive docs
appear at `/docs` (Swagger UI) and `/redoc`.
