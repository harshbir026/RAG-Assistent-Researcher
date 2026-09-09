# Pydantic as the validation layer

Pydantic models parse untrusted dicts/JSON into typed objects. FastAPI
uses them for bodies and for `response_model`. Standalone usage:

```python
from pydantic import BaseModel, Field, ValidationError

class Query(BaseModel):
    q: str = Field(min_length=3)
    limit: int = Field(default=10, ge=1, le=100)

try:
    Query(q="ab")
except ValidationError as e:
    print(e.errors())
```

v2 is faster (Rust core) and uses `model_validate`. Constraints belong on
the model so HTTP and CLI share one definition.
