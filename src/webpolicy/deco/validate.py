from collections.abc import Callable
from rich import print
from functools import wraps
from typing import Any

from pydantic import BaseModel, ValidationError


def inp(struct: type[BaseModel]) -> Callable[[Callable[[dict], Any]], Callable[[dict], Any]]:
    """Validate input dict against `struct` before running `fn`."""

    def _decorator(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if "x" in kwargs:
                payload = kwargs["x"]
            elif "obs" in kwargs:
                payload = kwargs["obs"]
            elif len(args) == 1:
                payload = args[0]
            elif len(args) >= 2:
                payload = args[1]
            else:
                raise TypeError("inp wrapper could not find input dict argument")

            try:
                struct.model_validate(payload)
            except ValidationError as e:
                print(e.errors())          # structured list
                raise e
            return fn(*args, **kwargs)

        return _wrapped

    return _decorator


def out(struct: type[BaseModel]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Validate function output against `struct` before returning it."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            try:
                struct.model_validate(result)
            except ValidationError as e:
                print(e.errors())          # structured list
                raise e
            return result

        return _wrapped

    return _decorator
