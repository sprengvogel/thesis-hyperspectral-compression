import torch.nn as nn

from compressai.layers import (
    conv3x3,
    subpel_conv3x3,
)


class ChengHyperprior(nn.Module):
    def __init__(self):
        super().__init__()
        N = 192
        self.encoder = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),  # Extra
            nn.LeakyReLU(inplace=True),  # Extra
            # conv3x3(N, N),
            conv3x3(N, N, stride=2),  # Extra
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            # conv3x3(N, N),
            conv3x3(N, N, stride=2),  # Extra
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.decoder = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),  # Extra
            nn.LeakyReLU(inplace=True),  # Extra
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            # conv3x3(N, N * 3 // 2),
            subpel_conv3x3(N, N * 3 // 2, 2),  # Extra
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            # conv3x3(N * 3 // 2, N * 2),
            subpel_conv3x3(N * 3 // 2, N * 2, 2),  # Extra
        )
