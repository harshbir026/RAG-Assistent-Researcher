# Response models and status codes

`response_model` filters the outbound JSON to declared fields. That prevents
accidentally leaking a hashed password from an ORM object.

```python
from fastapi import FastAPI, status
from pydantic import BaseModel

class UserOut(BaseModel):
    id: int
    email: str

app = FastAPI()

@app.post("/users/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user():
    return {"id": 1, "email": "a@b.com", "password_hash": "secret"}
```

The client receives `id` and `email` only. Use `status_code=` on the decorator
for the default success code. Raise `HTTPException(status_code=404, detail=...)`
for errors. FastAPI also supports `JSONResponse` if you need custom headers.
