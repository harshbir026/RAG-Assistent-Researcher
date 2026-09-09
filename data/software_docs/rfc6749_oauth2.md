# RFC 6749 — OAuth 2.0 grant types (summary)

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
