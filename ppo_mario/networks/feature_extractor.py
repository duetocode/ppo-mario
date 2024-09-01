import numpy as np
import torch
import torch.nn as nn
from .resent import AttentionResBlock
from .attention import Attention, SpatialGate
from gymnasium.spaces import Box, Space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResNetFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.extractor = nn.Sequential(
            AttentionResBlock(32, 64, stride=2),
            AttentionResBlock(64, 64, stride=2),
        )

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

    @property
    def attention_data(self):
        return [
            block.attention_data
            for block in self.extractor
            if hasattr(block, "attention_data")
        ]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # reshape the input to prepare and process the frames individually
        latent = obs.reshape(-1, 1, *obs.shape[-2:])
        latent = self.conv1(latent)

        # reshape the latent back to stacked frames for integrated processing
        latent = latent.reshape(-1, 32, *latent.shape[-2:])
        latent = self.extractor(latent)

        # project the latent to the feature dimension
        return self.projection(latent)


class AttentionCNN(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # these code are copied from stable_baselines3.common.torch_layers.NatureCNN
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # the attention layer
        self.attention = SpatialGate()

        # the flatten layer
        self.flatten = nn.Flatten()

        # the linear layer that projects the extracted features to the feature dimension
        with torch.no_grad():
            n_flatten = self.flatten(
                self.cnn(torch.as_tensor(observation_space.sample()[None]).float())
            ).shape[-1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # first, go through the CNN
        out = self.cnn(obs)
        # then, apply the attention layer
        out, scale = self.attention(out)
        # we save the attention for later use
        self.attention_data = [scale]
        # finally, flatten and project the features
        return self.linear(self.flatten(out))
