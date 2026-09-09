# Rate limiting

Rate limits protect capacity and reduce abuse. Token bucket and sliding
window are the usual algorithms. Advertise limits with `RateLimit-Limit`,
`RateLimit-Remaining`, and `Retry-After` on 429 responses.

Where to enforce: API gateway for coarse IP/app limits; application layer
for per-user and per-resource limits. Identity for the bucket should be
API key or user id, not only IP (NAT hides many users behind one address).
