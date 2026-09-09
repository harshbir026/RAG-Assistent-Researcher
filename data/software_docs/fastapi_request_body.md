# Request bodies and Pydantic models

POST/PUT/PATCH bodies are declared as Pydantic models. FastAPI parses JSON
into the model and returns 422 if the payload does not match.

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    tags: list[str] = []

app = FastAPI()

@app.post("/items/")
def create_item(item: Item):
    return item
```

A handler can mix a path parameter, a query parameter, and a body model in
one signature. FastAPI decides by source: path names match the URL, simple
types without path names are query params, Pydantic models are the body.
