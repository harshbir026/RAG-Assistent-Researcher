# Uvicorn and ASGI

ASGI is the async successor to WSGI. The server calls
`async def app(scope, receive, send)`. Uvicorn is an ASGI server that
speaks HTTP/1.1 and WebSockets. Gunicorn can manage Uvicorn workers:
`gunicorn -k uvicorn.workers.UvicornWorker main:app`.

Workers are processes. One Uvicorn worker with async handlers still uses
one event loop; CPU-bound work needs more workers or a process pool.
`--reload` is for development only.
