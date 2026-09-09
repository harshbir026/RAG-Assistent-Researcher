#!/usr/bin/env python3
"""Write original teaching notes into data/software_docs/. Run once from repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "software_docs"
ROOT.mkdir(parents=True, exist_ok=True)

DOCS: dict[str, str] = {}

DOCS["README.md"] = """# Software docs corpus (secondary)

Original teaching notes used to prove the RAG pipeline is corpus-agnostic.
These are **not** a dump of FastAPI's website. RFCs are summarized from the
public-domain IETF documents. FastAPI / Flask / HTTP notes are written from
first principles so the second collection stays legally and technically
separate from the arXiv research corpus.

Ingest with:

```
python src/ingest.py --corpus software_docs
python src/chunk.py --corpus software_docs
python src/embed.py --corpus software_docs
python src/evaluate.py --corpus software_docs --golden_set golden_qa_software.json
```
"""

DOCS["fastapi_first_steps.md"] = """# FastAPI first steps

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
"""

DOCS["fastapi_path_parameters.md"] = """# Path parameters and type validation in FastAPI

A path parameter is a variable segment of the URL. You declare it in the
decorator path and as a typed function argument. FastAPI reads the type hint,
converts the string from the URL, and rejects values that cannot convert.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

`GET /items/42` succeeds and `item_id` is the integer `42`. `GET /items/abc`
returns HTTP 422 with a validation error because `"abc"` is not an int.

To constrain the value further, wrap the hint with `Path`. Example: require
a positive id and document it in OpenAPI.

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
def read_user(user_id: int = Path(..., ge=1, description="Positive user id")):
    return {"user_id": user_id}
```

`Path(..., ge=1)` means the parameter is required and must be greater than
or equal to 1. `le`, `gt`, `lt`, `min_length`, and `max_length` work the same
way for numbers and strings.

Enum path parameters restrict the URL to a known set of values:

```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"

app = FastAPI()

@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

Order matters when two routes share a prefix: declare static paths such as
`/users/me` before `/users/{user_id}`, otherwise `"me"` is parsed as an id.

Interview takeaway: type validation is not a separate middleware you write.
It is the combination of the path template, the function annotation, and
Pydantic's conversion rules. That is how you define a path parameter with
type validation in FastAPI.
"""

DOCS["fastapi_query_parameters.md"] = """# Query parameters

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
"""

DOCS["fastapi_request_body.md"] = """# Request bodies and Pydantic models

POST/PUT/PATCH bodies are declared as Pydantic models. FastAPI parses JSON
into the model and returns 422 if the payload does not match.

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    tags: list[str] = []

app = FastAPI()

@app.post("/items/")
def create_item(item: Item):
    return item
```

A handler can mix a path parameter, a query parameter, and a body model in
one signature. FastAPI decides by source: path names match the URL, simple
types without path names are query params, Pydantic models are the body.
"""

DOCS["fastapi_response_models.md"] = """# Response models and status codes

`response_model` filters the outbound JSON to declared fields. That prevents
accidentally leaking a hashed password from an ORM object.

```python
from fastapi import FastAPI, status
from pydantic import BaseModel

class UserOut(BaseModel):
    id: int
    email: str

app = FastAPI()

@app.post("/users/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user():
    return {"id": 1, "email": "a@b.com", "password_hash": "secret"}
```

The client receives `id` and `email` only. Use `status_code=` on the decorator
for the default success code. Raise `HTTPException(status_code=404, detail=...)`
for errors. FastAPI also supports `JSONResponse` if you need custom headers.
"""

DOCS["fastapi_dependency_injection.md"] = """# FastAPI dependency injection

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
"""

DOCS["fastapi_async.md"] = """# FastAPI async request handling

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
"""

DOCS["fastapi_status_codes.md"] = """# HTTP status codes in APIs

Use 2xx for success (`200` default, `201` created, `204` no body). Use `400`
for malformed client input you do not validate with Pydantic (Pydantic
failures are `422` in FastAPI). `401` means unauthenticated; `403` means
authenticated but not allowed. `404` means the resource does not exist.
`409` is a conflict (duplicate create). `429` is rate limiting.

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/items/{item_id}", status_code=status.HTTP_200_OK)
def get_item(item_id: int):
    if item_id < 1:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```
"""

DOCS["fastapi_validation.md"] = """# Numeric and string validation

Path, query, and body validation share Pydantic constraints. `ge`/`le` bound
numbers. `min_length`/`max_length` bound strings. `regex` (or `pattern` in
Pydantic v2) bounds format. `Query(embed=True)` is only needed when you want
a query model nested as an object in OpenAPI.

Invalid values produce a 422 body with `loc`, `msg`, and `type` per error.
Do not catch those to turn them into 400 unless you have a compatibility
requirement; 422 is the framework's contract.
"""

DOCS["fastapi_middleware.md"] = """# Middleware and CORS

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
"""

DOCS["rfc7231_http_semantics.md"] = """# RFC 7231 — HTTP/1.1 semantics and content (summary)

RFC 7231 defines request methods, status codes, and payload semantics for
HTTP/1.1. It was later folded into RFC 9110; the status-code meanings below
are unchanged in spirit and are what interviewers mean by "per RFC 7231".

Safe methods (`GET`, `HEAD`, `OPTIONS`) should not change server state.
Idempotent methods (`GET`, `PUT`, `DELETE`, `HEAD`, `OPTIONS`) can be
retried: repeating the request has the same effect as doing it once.
`POST` is neither safe nor idempotent by default. `PATCH` is not required
to be idempotent.

Status codes 4xx mean the client appears to have erred. Two that are
routinely confused:

- **401 Unauthorized** — the request lacks valid authentication credentials
  for the target resource. The response should include a `WWW-Authenticate`
  challenge. The client may retry with credentials. Despite the word
  "Unauthorized" in the spec title, this is "you have not authenticated"
  (or the credentials were rejected).
- **403 Forbidden** — the server understood the request but refuses to
  authorize it. Authenticating will not help, or the authenticated user
  is not allowed. Do not send a credentials challenge as the primary
  meaning of 403.

Practical API mapping: missing/invalid token → 401. Valid token, wrong
role or resource ownership → 403. Unknown URL → 404 (do not leak whether
a hidden resource exists if that is a security requirement; some APIs
return 404 for both missing and forbidden).

Other 4xx worth knowing: 400 bad request syntax, 405 method not allowed,
409 conflict, 412 precondition failed, 415 unsupported media type, 429
too many requests (later RFCs / common practice).
"""

DOCS["rfc6749_oauth2.md"] = """# RFC 6749 — OAuth 2.0 grant types (summary)

OAuth 2.0 is a delegation protocol. A client obtains an *access token* so
it can call an API on behalf of a resource owner without seeing the owner's
password (except in the legacy password grant). RFC 6749 defines four
authorization grant types plus refresh.

1. **Authorization code.** The user is redirected to the authorization
   server, logs in, and the client receives a short-lived *code* at its
   redirect URI. The client then exchanges the code (plus client secret,
   for confidential clients) for an access token at the token endpoint.
   Use this for web apps and native apps (with PKCE, RFC 7636). This is
   the default grant for third-party login.

2. **Implicit.** The access token is returned directly in the redirect URI
   fragment. Designed for in-browser clients that could not keep a secret
   in 2012. **Do not use for new applications.** Authorization code + PKCE
   replaced it.

3. **Resource owner password credentials.** The client collects the user's
   username and password and posts them to the token endpoint. Only for
   first-party clients you fully trust, and even then it is discouraged
   because it trains users to type passwords into the client.

4. **Client credentials.** No user. The client authenticates as itself
   (client id + secret or assertion) and receives a token for
   application-level access. Use for service-to-service jobs, cron, and
   machine clients.

**Refresh tokens** are not a fifth way to obtain the first token; they are
a credential issued with some grants so the client can get a new access
token without repeating the user login. RFC 6749 also defines token
endpoint error codes (`invalid_grant`, `unauthorized_client`, etc.).

When to use each, in one line: third-party user login → authorization
code (+ PKCE). Browser SPA historically implicit, now code + PKCE. Trusted
first-party CLI that must use a password → password grant (avoid if you
can). Backend daemon → client credentials.
"""

DOCS["rfc6750_bearer.md"] = """# RFC 6750 — Bearer token usage

A bearer token is a credential that the holder can use without proving
possession of a second factor. RFC 6750 says how to send OAuth 2.0 access
tokens:

- `Authorization: Bearer <token>` (preferred)
- Form body `access_token=` when headers cannot be set
- Query `access_token=` as a last resort (leaks via logs and Referer)

If the token is missing or invalid, respond 401 with
`WWW-Authenticate: Bearer`. If the token is valid but lacks scope, 403 is
appropriate. Never log raw bearer tokens. TLS is mandatory; a stolen
bearer token is as powerful as a session cookie without HttpOnly caveats
on the wire.
"""

DOCS["rfc3986_uri.md"] = """# RFC 3986 — Uniform Resource Identifier

A URI has components: scheme, authority (userinfo, host, port), path,
query, and fragment. `https://api.example.com:443/v1/items?q=a#row` splits
into scheme `https`, host `api.example.com`, port `443`, path `/v1/items`,
query `q=a`, fragment `row`. Fragments are not sent to the server.

Percent-encoding protects reserved characters. Path parameters in web
frameworks are decoded path segments, not a different URI standard. When
designing APIs, keep resources in the path, filters in the query, and
never put secrets in the query string.
"""

DOCS["rfc9110_http.md"] = """# RFC 9110 — HTTP semantics (modern)

RFC 9110 consolidates HTTP semantics (including much of 7231). Methods,
status codes, content negotiation (`Accept`, `Content-Type`), conditional
requests (`ETag`, `If-None-Match`), and caching interplay are defined here
together with RFC 9111.

For API designers: treat method safety/idempotency as part of your public
contract. `GET` must be cacheable in intent. `PUT` replaces a resource at
a known URI. `DELETE` removes it. `POST` creates or triggers processing
when the client does not know the resulting URI. Conditional `If-Match`
prevents lost updates when two clients PUT the same resource.
"""

DOCS["api_design_versioning.md"] = """# API versioning

Three common strategies: URI prefix (`/v1/orders`), header
(`Accept: application/vnd.myapp.v1+json`), and no version with additive
evolution. URI versioning is the most discoverable for public HTTP APIs.
Header versioning keeps URLs stable. Additive evolution (only add optional
fields) works until a breaking change is unavoidable.

Breaking changes include renaming/removing fields, changing types, or
changing error codes. Additive optional fields and new endpoints are
usually non-breaking. Sunset old versions with a date and a header such as
`Deprecation`.
"""

DOCS["rest_idempotency.md"] = """# Idempotency in write APIs

Idempotency means repeating a request does not change the result beyond
the first successful application. `PUT /resources/{id}` with a full body
is naturally idempotent. `POST /charges` is not, unless you add an
`Idempotency-Key` header: the server stores the first response and returns
it for retries with the same key.

Without that key, a client timeout can double-charge. Keys should be unique
per logical operation, TTL-bounded (e.g. 24h), and stored with the response
status and body. Concurrent requests with the same key should wait or
serialize, not create two records.
"""

DOCS["rate_limiting.md"] = """# Rate limiting

Rate limits protect capacity and reduce abuse. Token bucket and sliding
window are the usual algorithms. Advertise limits with `RateLimit-Limit`,
`RateLimit-Remaining`, and `Retry-After` on 429 responses.

Where to enforce: API gateway for coarse IP/app limits; application layer
for per-user and per-resource limits. Identity for the bucket should be
API key or user id, not only IP (NAT hides many users behind one address).
"""

DOCS["pagination_patterns.md"] = """# Pagination

Offset pagination (`?offset=40&limit=20`) is simple and slow on large
tables (`OFFSET 100000` still scans). Cursor pagination (`?after=id_abc`)
is stable under inserts if the cursor is a unique ordered key. Return
`next_cursor` and avoid exposing total counts if counting is expensive.

Page numbers are offset in disguise. For realtime feeds, prefer cursors.
Always cap `limit` server-side.
"""

DOCS["caching_http.md"] = """# HTTP caching for APIs

`Cache-Control: private, max-age=60` lets a client reuse a GET. `ETag` plus
`If-None-Match` yields 304 without a body. Authenticated responses should
be `private` so shared proxies do not store them. `Vary: Authorization` is
a footgun (each token is a unique cache key); usually you just do not cache
personalized GETs at a CDN.

Mutating methods must not be cached. Use cache-busting on `POST` that
creates a resource the client will immediately GET.
"""

DOCS["database_indexing.md"] = """# Database indexing for API workloads

Index columns used in `WHERE`, `JOIN`, and `ORDER BY`. A composite index
`(user_id, created_at)` serves "list this user's recent orders". Selectivity
matters: an index on a boolean helps less than an index on a unique email.

Write cost: every index slows INSERT/UPDATE. Unused indexes are dead
weight. Pagination by indexed cursors avoids `OFFSET`. Measure with
`EXPLAIN ANALYZE`, not folklore.
"""

DOCS["queue_vs_sync.md"] = """# Sync HTTP vs background queues

If a request can finish in tens of milliseconds, do it in the handler. If
it emails, transcodes, or calls a slow partner, accept the request, enqueue
work, return 202 with a job URL. Workers pull from a queue (SQS, Redis,
RabbitMQ).

Idempotency keys belong on both the HTTP accept path and the worker: the
queue will deliver at least once. Timeouts on the HTTP side should not
equal "the work never ran".
"""

DOCS["observability.md"] = """# Observability for request-driven services

Logs, metrics, and traces answer different questions. Logs: what happened
on one request. Metrics: how often and how slow in aggregate. Traces: how
a request hopped across services.

Put a request id on every log line (`X-Request-ID`). RED metrics (rate,
errors, duration) per route. Trace the path from gateway → API → DB.
Do not log payloads that contain tokens or PII.
"""

DOCS["authn_vs_authz.md"] = """# Authentication vs authorization

Authentication answers who the caller is (password, OTP, OAuth token,
mTLS). Authorization answers what they may do (roles, permissions,
resource ownership). Mixing them produces the 401/403 confusion: 401 is
authn failure; 403 is authz failure after a successful authn.

Implement authn at the edge (verify JWT signature and expiry). Implement
authz in the application next to the resource (`order.user_id == token.sub`).
"""

DOCS["consistency_models.md"] = """# Consistency models APIs accidentally expose

A `GET` after a `POST` may not see the new row if you read from a replica
(read-your-writes violation). Strong consistency reads from the primary
or uses session affinity. Eventual consistency is fine for feeds; it is
not fine for "charge then show receipt" unless the receipt API reads the
write node.

Idempotent creates plus unique constraints are how you stay correct when
clients retry across replicas.
"""

DOCS["python_asyncio_notes.md"] = """# Python asyncio (notes)

The event loop runs one coroutine at a time until it `await`s. Concurrency
is cooperative. `asyncio.gather` runs several awaitables concurrently.
`asyncio.to_thread` offloads blocking callables.

```python
import asyncio

async def fetch_all():
    a, b = await asyncio.gather(get_a(), get_b())
    return a, b
```

Mixing asyncio with CPU-bound work needs a process pool; the loop does not
speed up math. Prefer one async style in a service: either ASGI + async
DB, or sync WSGI + threads, not both in the same request path.
"""

DOCS["flask_sync_vs_async.md"] = """# Flask synchronous request handling vs FastAPI async

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
"""

DOCS["pydantic_validation.md"] = """# Pydantic as the validation layer

Pydantic models parse untrusted dicts/JSON into typed objects. FastAPI
uses them for bodies and for `response_model`. Standalone usage:

```python
from pydantic import BaseModel, Field, ValidationError

class Query(BaseModel):
    q: str = Field(min_length=3)
    limit: int = Field(default=10, ge=1, le=100)

try:
    Query(q="ab")
except ValidationError as e:
    print(e.errors())
```

v2 is faster (Rust core) and uses `model_validate`. Constraints belong on
the model so HTTP and CLI share one definition.
"""

DOCS["uvicorn_asgi.md"] = """# Uvicorn and ASGI

ASGI is the async successor to WSGI. The server calls
`async def app(scope, receive, send)`. Uvicorn is an ASGI server that
speaks HTTP/1.1 and WebSockets. Gunicorn can manage Uvicorn workers:
`gunicorn -k uvicorn.workers.UvicornWorker main:app`.

Workers are processes. One Uvicorn worker with async handlers still uses
one event loop; CPU-bound work needs more workers or a process pool.
`--reload` is for development only.
"""

DOCS["httpx_vs_requests.md"] = """# httpx vs requests

`requests` is sync-only. `httpx` offers sync and async clients with the
same API shape. In FastAPI `async def` handlers, use `httpx.AsyncClient`
(shared via a dependency, not created per request if you can help it).
In Flask sync views, `requests` is fine.

Connection pooling matters more than the library name. Do not open a new
client per request in a tight loop. Timeouts must be set; the default
"wait forever" is a production incident.
"""


def main() -> None:
    for name, body in DOCS.items():
        path = ROOT / name
        path.write_text(body.strip() + "\n", encoding="utf-8")
        print(f"wrote {path}")
    md_files = [p for p in ROOT.glob("*.md") if p.name != "README.md"]
    print(f"\n{len(md_files)} teaching notes (README excluded from chunker)")


if __name__ == "__main__":
    main()
