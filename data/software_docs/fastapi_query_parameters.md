# Query parameters

Query parameters live after `?` in the URL. Any function argument that is
not a path parameter becomes a query parameter.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10, q: str | None = None):
    return {"skip": skip, "limit": limit, "q": q}
```

`GET /items/?skip=20&limit=5` binds `skip=20`, `limit=5`, `q=None`. Optional
parameters use a default (`None` or a value). Required query parameters have
no default.

Validate with `Query`: `q: str | None = Query(None, min_length=3, max_length=50)`.
Booleans accept `true`, `false`, `1`, `0`, `on`, `off`.
