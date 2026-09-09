# Observability for request-driven services

Logs, metrics, and traces answer different questions. Logs: what happened
on one request. Metrics: how often and how slow in aggregate. Traces: how
a request hopped across services.

Put a request id on every log line (`X-Request-ID`). RED metrics (rate,
errors, duration) per route. Trace the path from gateway → API → DB.
Do not log payloads that contain tokens or PII.
