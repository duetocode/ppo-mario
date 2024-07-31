import torch
import numpy as np
from ppo_mario.networks import ResNetFeatureExtractor
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3.ppo import PPO


class DummyEnv(Env):

    def __init__(self, *args, **kwargs):
        self.action_space = Discrete(4)
        self.observation_space = Box(0, 255, (8, 104, 128), dtype=np.uint8)

    def step(self, *args, **kwargs):
        return None, 0, False, False, {}

    def reset(self, *args, **kwargs):
        return None, {}


def test_feature_extractor():
    extractor = ResNetFeatureExtractor(Box(0, 255, (8, 104, 128)))
    data = np.random.randint(0, 256, (3, 8, 104, 128), dtype=np.uint8) / 255.0

    with torch.no_grad():
        result = extractor(torch.as_tensor(data, dtype=torch.float))

    assert result.shape == (3, 256)


def test_load_model():
    model = PPO.load(
        "assets/model-supervised.zip",
    )

    data = np.random.randint(0, 255, (3, 8, 104, 128), dtype=np.uint8)
    result = model.predict(data)

    assert result is not None
