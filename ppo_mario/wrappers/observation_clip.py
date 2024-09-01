from typing import Tuple
import gymnasium
from gymnasium import Env, spaces
from gymnasium import ObservationWrapper
import numpy as np


class ObservationClip(ObservationWrapper):
    """Wrapper that clips the observation to hide to top part of the screen, which contains the timers and the score. These are irrelevant information that may interfere the training."""

    def __init__(self, env: Env, top: int = 0, left: int = 0):
        super().__init__(env)

        # ensure the observation space is gym.spaces.Box
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError(
                "ObservationClip only works with gym.spaces.Box observation spaces."
            )

        self.observation_space = spaces.Box(0, 255, (120 - 16, 128), dtype=np.uint8)

    def observation(self, obs):
        return obs[16:, :]
