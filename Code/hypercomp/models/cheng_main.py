import torch.nn as nn

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


class ChengMain(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        N = 192
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, N),  # Extra
            # ResidualBlockWithStride(in_channels, N, stride=2),
            ResidualBlock(N, N),  # Extra
            # ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),  # Extra
            # ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            # conv3x3(N, N),  # Extra
            AttentionBlock(N),
        )

        self.decoder = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),  # Extra
            # ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),  # Extra
            # ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),  # Extra
            # ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, in_channels, 2),
            # subpel_conv3x3(N, in_channels),  # Extra
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
