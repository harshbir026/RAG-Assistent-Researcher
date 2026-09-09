# Path parameters and type validation in FastAPI

A path parameter is a variable segment of the URL. You declare it in the
decorator path and as a typed function argument. FastAPI reads the type hint,
converts the string from the URL, and rejects values that cannot convert.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

`GET /items/42` succeeds and `item_id` is the integer `42`. `GET /items/abc`
returns HTTP 422 with a validation error because `"abc"` is not an int.

To constrain the value further, wrap the hint with `Path`. Example: require
a positive id and document it in OpenAPI.

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
def read_user(user_id: int = Path(..., ge=1, description="Positive user id")):
    return {"user_id": user_id}
```

`Path(..., ge=1)` means the parameter is required and must be greater than
or equal to 1. `le`, `gt`, `lt`, `min_length`, and `max_length` work the same
way for numbers and strings.

Enum path parameters restrict the URL to a known set of values:

```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"

app = FastAPI()

@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

Order matters when two routes share a prefix: declare static paths such as
`/users/me` before `/users/{user_id}`, otherwise `"me"` is parsed as an id.

Interview takeaway: type validation is not a separate middleware you write.
It is the combination of the path template, the function annotation, and
Pydantic's conversion rules. That is how you define a path parameter with
type validation in FastAPI.
