import dataclasses as dc
from pathlib import Path
import json


@dc.dataclass
class TrainConfiguration:
    batch_size: int = 128
    n_epochs: int = 8
    n_steps: int = 2048
    gamma: float = 0.9
    learning_rate: float = 1e-4
    freeze_actor: bool = False
    target_kl: float = 0.2
    clip_range: float = 0.1
    total_timesteps: int = 500_000
    normalize_advantage: bool = False
    random_frame_skip: bool = False

    def to_json(self) -> str:
        return json.dumps(dc.asdict(self), indent=4)

    @classmethod
    def load(self, encoded: str) -> "TrainConfiguration":
        return TrainConfiguration(**json.loads(encoded))

    @property
    def ppo_cfg(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            n_steps=self.n_steps,
            verbose=0,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            target_kl=self.target_kl,
            clip_range=self.clip_range,
            normalize_advantage=self.normalize_advantage,
        )
