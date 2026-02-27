import pytest
from pydantic import BaseModel, ValidationError

from webpolicy.deco.validate import inp, out


class User(BaseModel):
    id: int
    name: str


def test_inp_allows_pydantic_coercion_and_calls_fn_with_original_dict():
    seen = {}

    @inp(User)
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


def test_inp_raises_on_invalid_payload():
    @inp(User)
    def fn(x: dict) -> dict:
        return x

    with pytest.raises(ValidationError):
        fn({"id": "abc", "name": "Alice"})


def test_out_allows_valid_output():
    @out(User)
    def fn() -> dict:
        return {"id": "123", "name": "Alice"}

    assert fn() == {"id": "123", "name": "Alice"}


def test_out_raises_on_invalid_output():
    @out(User)
    def fn() -> dict:
        return {"id": "abc", "name": "Alice"}

    with pytest.raises(ValidationError):
        fn()
