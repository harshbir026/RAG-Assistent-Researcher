# Pagination

Offset pagination (`?offset=40&limit=20`) is simple and slow on large
tables (`OFFSET 100000` still scans). Cursor pagination (`?after=id_abc`)
is stable under inserts if the cursor is a unique ordered key. Return
`next_cursor` and avoid exposing total counts if counting is expensive.

Page numbers are offset in disguise. For realtime feeds, prefer cursors.
Always cap `limit` server-side.
