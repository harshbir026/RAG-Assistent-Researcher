# Python asyncio (notes)

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
