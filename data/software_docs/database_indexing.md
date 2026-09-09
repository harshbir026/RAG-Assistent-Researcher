# Database indexing for API workloads

Index columns used in `WHERE`, `JOIN`, and `ORDER BY`. A composite index
`(user_id, created_at)` serves "list this user's recent orders". Selectivity
matters: an index on a boolean helps less than an index on a unique email.

Write cost: every index slows INSERT/UPDATE. Unused indexes are dead
weight. Pagination by indexed cursors avoids `OFFSET`. Measure with
`EXPLAIN ANALYZE`, not folklore.
