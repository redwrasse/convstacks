import torch

from wavenetlike.oneconv import ReluOneConv


class SkipPostProcessing(torch.nn.Module):

    """ two compositions of relu followed by 1 conv"""

    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(SkipPostProcessing, self).__init__()
        self.relu_conv_a = ReluOneConv(
            in_channels=in_channels,
            out_channels=intermediate_channels
        )
        self.relu_conv_b = ReluOneConv(
            in_channels=intermediate_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        out_a = self.relu_conv_a(x)
        out_b = self.relu_conv_b(out_a)
        return out_b
