# RFC 7231 — HTTP/1.1 semantics and content (summary)

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
