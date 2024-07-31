import numpy as np
import torch
import torch.nn as nn
from .resent import ResBlock
from .attention import PoolingAttention
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResNetFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
            ResBlock(4, 4, stride=2),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.extractor = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
        )

        self.attention2 = PoolingAttention()

        # calculate the output dimensions
        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None] / 255, dtype=torch.float
            )
            sample = sample.view(-1, 1, *sample.shape[-2:])
            sample = self.conv1(sample)
            sample = sample.view(-1, 32, *sample.shape[-2:])
            latent_dims = np.prod(self.extractor(sample).shape[1:])

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dims, features_dim, bias=False),
            nn.BatchNorm1d(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # reshape the input to prepare and process the frames individually
        latent = obs.reshape(-1, 1, *obs.shape[-2:])
        latent = self.conv1(latent)
        # apply the first attention layer
        attention = self.attention(latent)
        latent = latent * attention
        # reshape the latent back to stacked frames for integrated processing
        latent = latent.reshape(-1, 32, *latent.shape[-2:])
        latent = self.extractor(latent)

        # apply the second attention layer
        attention = self.attention2(latent)
        self._attention = attention
        latent = latent * attention

        # project the latent to the feature dimension
        return self.projection(latent)
