import torch
import torch.nn as nn

class PoolingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
            
    def forward(self, x):
        latent = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0],
        ], dim=1)
        return self.attention(latent)