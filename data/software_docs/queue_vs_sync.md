# Sync HTTP vs background queues

If a request can finish in tens of milliseconds, do it in the handler. If
it emails, transcodes, or calls a slow partner, accept the request, enqueue
work, return 202 with a job URL. Workers pull from a queue (SQS, Redis,
RabbitMQ).

Idempotency keys belong on both the HTTP accept path and the worker: the
queue will deliver at least once. Timeouts on the HTTP side should not
equal "the work never ran".
