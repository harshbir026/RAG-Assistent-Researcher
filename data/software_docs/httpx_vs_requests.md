# httpx vs requests

`requests` is sync-only. `httpx` offers sync and async clients with the
same API shape. In FastAPI `async def` handlers, use `httpx.AsyncClient`
(shared via a dependency, not created per request if you can help it).
In Flask sync views, `requests` is fine.

Connection pooling matters more than the library name. Do not open a new
client per request in a tight loop. Timeouts must be set; the default
"wait forever" is a production incident.
