import pytest
from pydantic import BaseModel, ValidationError

from webpolicy.deco.validate import validate


class User(BaseModel):
    id: int
    name: str


def test_validate_allows_pydantic_coercion_and_calls_fn_with_original_dict():
    seen = {}

    @validate(User)
    def fn(x: dict) -> tuple[str, str]:
        seen["id_type"] = type(x["id"]).__name__
        return str(x["id"]), x["name"]

    data = {"id": "123", "name": "Alice"}
    out = fn(data)

    assert out == ("123", "Alice")
    assert seen["id_type"] == "str"

    user = User(**data)
    assert user.id == 123
    assert type(user.id) is int


def test_validate_raises_on_invalid_payload():
    @validate(User)
    def fn(x: dict) -> dict:
        return x

    with pytest.raises(ValidationError):
        fn({"id": "abc", "name": "Alice"})
