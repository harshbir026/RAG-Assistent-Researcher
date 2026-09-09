# Idempotency in write APIs

Idempotency means repeating a request does not change the result beyond
the first successful application. `PUT /resources/{id}` with a full body
is naturally idempotent. `POST /charges` is not, unless you add an
`Idempotency-Key` header: the server stores the first response and returns
it for retries with the same key.

Without that key, a client timeout can double-charge. Keys should be unique
per logical operation, TTL-bounded (e.g. 24h), and stored with the response
status and body. Concurrent requests with the same key should wait or
serialize, not create two records.
