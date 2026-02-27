import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def step(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self, payload: Dict | None = None) -> None:
        """Reset the policy to its initial state.

        Args:
            payload: Optional reset payload (defaults to {} behaviorally).
        """
        pass
