# Consistency models APIs accidentally expose

A `GET` after a `POST` may not see the new row if you read from a replica
(read-your-writes violation). Strong consistency reads from the primary
or uses session affinity. Eventual consistency is fine for feeds; it is
not fine for "charge then show receipt" unless the receipt API reads the
write node.

Idempotent creates plus unique constraints are how you stay correct when
clients retry across replicas.
