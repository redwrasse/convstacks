import torch


class OneConv(torch.nn.Conv1d):

    """ a '1 conv' """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):  # bias needed?
        super(OneConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        return super(OneConv, self).forward(x)


class ReluOneConv(torch.nn.Module):
    """ relu followed by one conv for skip post-processing """

    def __init__(self, in_channels, out_channels):
        super(ReluOneConv, self).__init__()
        self.one_conv = OneConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        relu_x = torch.relu(x)
        out = self.one_conv(relu_x)
        return out