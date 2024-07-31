from typing import Any
from pathlib import Path
import random
import lzma
from gymnasium import Wrapper, Env


class RandomEpisode(Wrapper):
    """Begin the episode with a random checkpoint."""

    def __init__(
        self,
        env: Env,
        data_dir: str | Path,
        seed: int = None,
        random_starts_rate: float = 0.5,
    ):
        super().__init__(env)

        data_dir = Path(data_dir)
        if not (data_dir.exists() and data_dir.is_dir()):
            raise ValueError(f"Invalid data directory: {str(data_dir)}")

        self.random_starts_rate = random_starts_rate

        self._rng = random.Random(seed)

        # iterate the checkpoints
        self._checkpoints = sorted(data_dir.glob("*.state.xz"))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # reset the environment
        obs, info = self.env.reset(seed=seed, options=options)

        # randomly select a checkpoint with the rate
        if self._rng.random() < self.random_starts_rate:
            checkpoint = self._rng.choice(self._checkpoints)
            # load the checkpoint
            saved_state = lzma.decompress(checkpoint.read_bytes())
            self.env.unwrapped.deserialize(saved_state)
            # refresh the info because the state is changed
            info = self.get_info()

        return obs, info
