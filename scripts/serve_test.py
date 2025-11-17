import dataclasses
import logging
import enum
import time

from rich.pretty import pprint
import numpy as np
import tyro

from webpolicy.base_policy import BasePolicy
from webpolicy.server import WebsocketPolicyServer


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # env: EnvMode = EnvMode.ALOHA_SIM

    num_steps: int = 10


class DummyPolicy(BasePolicy):

    def infer(self, obs: dict) -> dict:
        """Dummy inference function that returns the observation."""
        pass

    def reset(self, *args, **kwargs) -> None:
        """Dummy reset function."""
        pass


def main(cfg: Args) -> None:
    pprint(cfg)

    policy = DummyPolicy()
    server = WebsocketPolicyServer(
        policy=policy,
        host=cfg.host,
        port=cfg.port,
        # metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
