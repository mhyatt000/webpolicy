import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from webpolicy.base_policy import BasePolicy
from webpolicy.deco.validate import inp


class ObsModel(BaseModel):
    id: int
    name: str


class DummyPolicy(BasePolicy):
    @inp(ObsModel)
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
