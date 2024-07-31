import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class DummyEnv(Env):

    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=255, shape=(8, 104, 128), dtype=np.uint8
        )
        self.action_space = Discrete(4)
