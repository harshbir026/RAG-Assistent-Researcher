# Numeric and string validation

Path, query, and body validation share Pydantic constraints. `ge`/`le` bound
numbers. `min_length`/`max_length` bound strings. `regex` (or `pattern` in
Pydantic v2) bounds format. `Query(embed=True)` is only needed when you want
a query model nested as an object in OpenAPI.

Invalid values produce a 422 body with `loc`, `msg`, and `type` per error.
Do not catch those to turn them into 400 unless you have a compatibility
requirement; 422 is the framework's contract.
