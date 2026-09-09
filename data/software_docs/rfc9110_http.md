# RFC 9110 — HTTP semantics (modern)

RFC 9110 consolidates HTTP semantics (including much of 7231). Methods,
status codes, content negotiation (`Accept`, `Content-Type`), conditional
requests (`ETag`, `If-None-Match`), and caching interplay are defined here
together with RFC 9111.

For API designers: treat method safety/idempotency as part of your public
contract. `GET` must be cacheable in intent. `PUT` replaces a resource at
a known URI. `DELETE` removes it. `POST` creates or triggers processing
when the client does not know the resulting URI. Conditional `If-Match`
prevents lost updates when two clients PUT the same resource.
