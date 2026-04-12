import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super().__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=True),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.size()

        avg_pool = torch.mean(x, dim=(2,3)).view(b,c)
        max_pool = torch.amax(x, dim=(2,3)).view(b,c)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        out = avg_out + max_out

        out = self.sigmoid(out).view(b,c,1,1)

        return x * out


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool,_ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_pool, max_pool], dim=1)

        out = self.conv(concat)

        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x