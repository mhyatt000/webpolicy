import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def step(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
