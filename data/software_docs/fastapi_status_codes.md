# HTTP status codes in APIs

Use 2xx for success (`200` default, `201` created, `204` no body). Use `400`
for malformed client input you do not validate with Pydantic (Pydantic
failures are `422` in FastAPI). `401` means unauthenticated; `403` means
authenticated but not allowed. `404` means the resource does not exist.
`409` is a conflict (duplicate create). `429` is rate limiting.

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/items/{item_id}", status_code=status.HTTP_200_OK)
def get_item(item_id: int):
    if item_id < 1:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```
