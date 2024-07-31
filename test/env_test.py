import numpy as np
from ppo_mario.train import create_env


def test_env():
    env = create_env()

    assert env.observation_space.dtype == np.uint8

    assert env is not None
