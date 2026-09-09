# Middleware and CORS

Middleware wraps every request. FastAPI (Starlette) middleware is ASGI:
`def dispatch(self, request, call_next)`. Common uses: request IDs, timing
headers, CORS.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization"],
)
```

CORS is a browser rule, not an API-security control. Server-to-server
clients ignore it. Auth still belongs in dependencies, not in CORS config.
