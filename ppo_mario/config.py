import dataclasses as dc
from pathlib import Path
import json
from typing import Type


@dc.dataclass
class TrainConfiguration:
    # batch size for the PPO
    batch_size: int = 128
    # number of epochs for each PPO update
    n_epochs: int = 8
    # the number of steps for each rollout
    n_steps: int = 2048
    # the discount factor for reward rollout
    gamma: float = 0.9
    # the learning rate for the PPO
    learning_rate: float = 1e-4
    # whether to freeze the actor-related network
    freeze_actor: bool = False
    # the target KL divergence for the PPO
    target_kl: float = 0.2
    # the clip range for the PPO
    clip_range: float = 0.2
    # the total number of timesteps for the training
    total_timesteps: int = 500_000
    # whether to normalize the advantage, for the PPO
    normalize_advantage: bool = False
    # whether to skip frames randomly
    random_frame_skip: bool = True
    # the value function coefficient for the PPO
    vf_coef: float = 0.5
    # the entropy coefficient for the PPO, which is the exploration factor
    ent_coef: float = 0.0
    # the level to play
    level: tuple = (4, 1)
    # the arguments for the policy network of the PPO model
    policy_kwargs: dict = dc.field(default_factory=dict)
    # the reward parameters for the custom reward scheme
    reward_params: dict = dc.field(default_factory=dict)

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
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
        )
