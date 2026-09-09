# RFC 6750 — Bearer token usage

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
