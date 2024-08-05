import dataclasses as dc
from pathlib import Path
import json


@dc.dataclass
class TrainConfiguration:
    work_dir: str
    base_model: str | None = None
    device: str | None = None
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

    @property
    def logs_dir(self) -> Path:
        return Path(self.work_dir, "logs")

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.work_dir, "checkpoints")

    @property
    def saved_model(self) -> Path:
        return Path(self.work_dir, "model.zip")

    def save(self) -> str:
        data = self.to_json()
        Path(self.work_dir, "config.json").write_text(data)
        return data

    def prepare_work_dir(self):
        # fail if the directory already exists
        Path(self.work_dir).mkdir(parents=True, exist_ok=False)
        # other children
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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
