import torch
import torch.nn as nn
import torch.nn.functional as F

"""
@misc{woo2018cbamconvolutionalblockattention,
    title={CBAM: Convolutional Block Attention Module},
    author={Sanghyun Woo and Jongchan Park and Joon-Young Lee and In So Kweon},
    year={2018},
    eprint={1807.06521},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/1807.06521},
}
"""


class ChannelGate(nn.Module):
    def __init__(self, n_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels, n_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(n_channels // reduction_ratio, n_channels),
        )

    def forward(self, x):
        channel_att_sum = None
        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att_sum = self.mlp(avg_pool) + self.mlp(max_pool)

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = torch.cat(
            [
                torch.mean(x, dim=1, keepdim=True),
                torch.max(x, dim=1, keepdim=True)[0],
            ],
            dim=1,
        )
        x_out = self.spatial(latent)
        scale = F.sigmoid(x_out)
        return x * scale, scale


class Attention(nn.Module):

    def __init__(self, n_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_gate = ChannelGate(n_channels, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out, scale = self.spatial_gate(x)
        # return the scale for further visualization
        return x_out, scale
