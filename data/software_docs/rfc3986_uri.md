# RFC 3986 — Uniform Resource Identifier

A URI has components: scheme, authority (userinfo, host, port), path,
query, and fragment. `https://api.example.com:443/v1/items?q=a#row` splits
into scheme `https`, host `api.example.com`, port `443`, path `/v1/items`,
query `q=a`, fragment `row`. Fragments are not sent to the server.

Percent-encoding protects reserved characters. Path parameters in web
frameworks are decoded path segments, not a different URI standard. When
designing APIs, keep resources in the path, filters in the query, and
never put secrets in the query string.
