# API versioning

Three common strategies: URI prefix (`/v1/orders`), header
(`Accept: application/vnd.myapp.v1+json`), and no version with additive
evolution. URI versioning is the most discoverable for public HTTP APIs.
Header versioning keeps URLs stable. Additive evolution (only add optional
fields) works until a breaking change is unavoidable.

Breaking changes include renaming/removing fields, changing types, or
changing error codes. Additive optional fields and new endpoints are
usually non-breaking. Sunset old versions with a date and a header such as
`Deprecation`.
