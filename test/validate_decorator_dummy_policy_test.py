import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from typing import Any
from webpolicy.base_policy import BasePolicy
from webpolicy.deco import validate 


class ObsModel(BaseModel):
    id: int
    name: str

class ResultModel(BaseModel):
    echo_name: str
    echo_id_type: str
    action: Any


class DummyPolicy(BasePolicy):
    @validate.inp(ObsModel)
    @validate.out(ResultModel)
    def step(self, obs: dict) -> dict:
        # Uses the original dict passed into step.
        return {
            "echo_name": obs["name"],
            "echo_id_type": type(obs["id"]).__name__,
            "action": np.array([1], dtype=np.int32),
        }

    def reset(self) -> None:
        pass


def test_dummy_policy_step_with_inp_decorator_accepts_coercible_input():
    policy = DummyPolicy()
    obs = {"id": "123", "name": "Alice"}

    action = policy.step(obs)

    assert action["echo_name"] == "Alice"
    assert action["echo_id_type"] == "str"
    np.testing.assert_array_equal(action["action"], np.array([1], dtype=np.int32))


def test_dummy_policy_step_with_inp_decorator_rejects_invalid_input():
    policy = DummyPolicy()

    with pytest.raises(ValidationError):
        policy.step({"id": "abc", "name": "Alice"})


def test_dummy_policy_step_with_out_decorator_rejects_invalid_output():
    class InvalidOutputPolicy(BasePolicy):
        @validate.inp(ObsModel)
        @validate.out(ResultModel)
        def step(self, obs: dict) -> dict:
            return {
                "echo_name": obs["name"],
                "echo_id_type": type(obs["id"]).__name__,
            }

        def reset(self) -> None:
            pass

    policy = InvalidOutputPolicy()
    obs = {"id": 123, "name": "Alice"}

    with pytest.raises(ValidationError):
        policy.step(obs)
