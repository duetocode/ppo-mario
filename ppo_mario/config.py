import dataclasses as dc
from pathlib import Path
import json
from typing import Type


@dc.dataclass
class TrainConfiguration:
    batch_size: int = 128
    n_epochs: int = 8
    n_steps: int = 2048
    gamma: float = 0.9
    learning_rate: float = 1e-4
    freeze_actor: bool = False
    target_kl: float = 0.2
    clip_range: float = 0.2
    total_timesteps: int = 500_000
    normalize_advantage: bool = False
    random_frame_skip: bool = True
    level: tuple = (4, 1)
    policy_kwargs: dict = dc.field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the configuration object to a JSON string"""
        return json.dumps(dc.asdict(self), indent=4)

    @classmethod
    def load(self, encoded: str) -> "TrainConfiguration":
        """
        Load the configuration object from a JSON string

        Parameters
        ----------
        encoded : str
            The JSON string.

        Returns
        -------
        TrainConfiguration
            The configuration object
        """
        return TrainConfiguration(**json.loads(encoded))

    @property
    def ppo_cfg(self) -> dict:
        """Get the kwargs fro the PPO class."""

        return dict(
            batch_size=self.batch_size,
            n_steps=self.n_steps,
            verbose=0,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            target_kl=self.target_kl,
            clip_range=self.clip_range,
            normalize_advantage=self.normalize_advantage,
            policy_kwargs=self.policy_kwargs,
        )
