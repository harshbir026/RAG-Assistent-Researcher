# HTTP caching for APIs

`Cache-Control: private, max-age=60` lets a client reuse a GET. `ETag` plus
`If-None-Match` yields 304 without a body. Authenticated responses should
be `private` so shared proxies do not store them. `Vary: Authorization` is
a footgun (each token is a unique cache key); usually you just do not cache
personalized GETs at a CDN.

Mutating methods must not be cached. Use cache-busting on `POST` that
creates a resource the client will immediately GET.
