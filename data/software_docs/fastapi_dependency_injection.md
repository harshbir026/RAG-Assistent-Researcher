# FastAPI dependency injection

FastAPI's dependency injection system lets you declare reusable units of work
(database sessions, auth, pagination) as functions, then attach them to path
operations with `Depends`. FastAPI builds a dependency graph, calls each
dependency with the current request, caches results per request, and injects
the return value into the handler.

```python
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()

def get_db():
    db = {"connected": True}
    try:
        yield db
    finally:
        db["connected"] = False

def require_token(x_token: str = Header(...)):
    if x_token != "secret":
        raise HTTPException(status_code=401, detail="Invalid token")
    return x_token

@app.get("/secure-items")
def read_secure_items(
    db=Depends(get_db),
    token: str = Depends(require_token),
):
    return {"db": db["connected"], "token": token}
```

How it works, step by step:

1. **Declaration.** A dependency is any callable. It can take the same kinds
   of parameters as a path operation: query, header, cookie, or nested
   `Depends`.
2. **Resolution.** Before the path function runs, FastAPI calls
   `require_token` and `get_db`. Nested dependencies are solved first.
3. **Generator cleanup.** If the dependency `yield`s, the body of the path
   function runs at the yield point. Code after `yield` runs after the
   response is sent (or after an exception), which is how you close a DB
   session.
4. **Caching.** Two handlers that `Depends(get_db)` in the same request share
   one `get_db` call unless you set `use_cache=False`.
5. **Overrides.** Tests replace `get_db` with `app.dependency_overrides[get_db] = fake_db`.

Sub-dependencies compose: `get_current_user` can depend on `require_token`,
and a path can depend on `get_current_user` without repeating header parsing.

This is not Flask's `g` object and not a global singleton. It is request-scoped
wiring driven by type hints, which is why the same pattern scales from a
single header check to a full auth + tenancy + unit-of-work stack.
