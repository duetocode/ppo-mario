import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
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

    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False),
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
        return self.attention(latent)
