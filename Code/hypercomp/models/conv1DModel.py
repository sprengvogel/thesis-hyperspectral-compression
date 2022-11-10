import torch
from torch.nn import LeakyReLU
from .. import params as p


class Conv1DModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Conv1DEncoder()
        self.decoder = Conv1DDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Conv1DEncoder(torch.nn.Module):

    def __init__(self, input_channels) -> None:
        super().__init__()
        if not input_channels % 4 == 0:
            raise ValueError(
                "Channels have to be divisible by 4. Consider padding.")
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_channels, out_channels=input_channels, kernel_size=11)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(
            in_channels=input_channels//2, out_channels=input_channels//2, kernel_size=11)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv3 = torch.nn.Conv1d(
            in_channels=input_channels//4, out_channels=input_channels//4, kernel_size=9)
        self.conv4 = torch.nn.Conv1d(
            in_channels=input_channels//4, out_channels=input_channels//4, kernel_size=7)

    def forward(self, x):
        out = self.maxpool1(LeakyReLU(self.conv1(x)))
        out = self.maxpool2(LeakyReLU(self.conv2(out)))
        out = LeakyReLU(self.conv3(out))
        return LeakyReLU(self.conv4(out))


class Conv1DDecoder(torch.nn.Module):

    def __init__(self, output_channels) -> None:
        super().__init__()
        if not output_channels % 4 == 0:
            raise ValueError(
                "Channels have to be divisible by 4. Consider padding.")
        self.conv1 = torch.nn.Conv1d(
            in_channels=output_channels//4, out_channels=output_channels//4, kernel_size=7)
        self.conv2 = torch.nn.Conv1d(
            in_channels=output_channels//4, out_channels=output_channels//4, kernel_size=9)
        self.upsampling1 = torch.nn.Upsample(scale_factor=2)
        self.conv3 = torch.nn.Conv1d(
            in_channels=output_channels//2, out_channels=output_channels//2, kernel_size=11)
        self.upsampling2 = torch.nn.Upsample(scale_factor=2)
        self.conv4 = torch.nn.Conv1d(
            in_channels=output_channels, out_channels=output_channels, kernel_size=11)

    def forward(self, x):
        out = self.upsampling1(LeakyReLU(self.conv1(x)))
        out = self.upsampling2(LeakyReLU(self.conv2(out)))
        out = LeakyReLU(self.conv3(out))
        return torch.nn.Sigmoid(self.conv4(out))
