# building_blocks.py

import torch
from torch.nn import functional as F


class LpConv(torch.nn.Conv1d):
    """
    A left-padded convolution
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):  # no convolution biases in wavenet
        super(LpConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # from pytorch docs: output length =
        # (input length + 2 * padding - dilation * (kernel_size - 1) -1 ) / stride + 1

    def forward(self, x):
        input_length = x.shape[-1]
        # note self.padding parameter is conv1d double padding parameter, not our padding
        output_length = \
            int((input_length + 2 * self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1) - 1) / self.stride[0] + 1)
        # set additional padding to ensure output length = input length
        __padding = input_length - output_length
        lp_x = F.pad(x,
                     pad=[__padding, 0],
                     mode='constant',
                     value=0)
        return super(LpConv, self).forward(lp_x)


class Block:
    """
        A block is a composition of increasingly dilated layers

        :param n_layers:
        :param kernel_length:
        :param dilation_rate:
        :param in_channels:
        :param out_channels:
        :param intermediate_channels:
        :return:
        """
    def __init__(self, n_layers, kernel_length, dilation_rate=2, in_channels=1,
                 out_channels=1, intermediate_channels=1):
        self.n_layers = n_layers
        self.kernel_length = kernel_length
        self.dilation_rate = dilation_rate
        self.model = _build_block(n_layers=n_layers,
                                  kernel_length=kernel_length,
                                  dilation_rate=dilation_rate,
                                  in_channels=in_channels,
                                  out_channels=out_channels,
                                  intermediate_channels=intermediate_channels)


def _build_block(n_layers, kernel_length, dilation_rate, in_channels,
                 out_channels, intermediate_channels):
    """
    A block is a composition of increasingly dilated layers

    :param n_layers:
    :param kernel_length:
    :param dilation_rate:
    :param in_channels:
    :param out_channels:
    :param intermediate_channels:
    :return:
    """
    # to do: generalize parameters
    # effective kernel length with stride of 1 for n layers is
    # sum_{i=0 to n-1} kernel_length * dilation_rate**i
    lpc = LpConv(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=kernel_length,
                 dilation=dilation_rate**0)
    for i in range(n_layers-2):
        lpc = torch.nn.Sequential(lpc, LpConv(in_channels=intermediate_channels,
                                              out_channels=intermediate_channels,
                                              kernel_size=kernel_length,
                                              dilation=dilation_rate**(i+1)))
    lpc = torch.nn.Sequential(lpc, LpConv(in_channels=intermediate_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_length,
                                              dilation=dilation_rate**(n_layers-1)))
    return lpc


class GatedActivationUnit(torch.nn.Module):

    """ Gated activation unit from wavenet """

    def __init__(self):
        super(GatedActivationUnit, self).__init__()

    def forward(self, wf, wg):
        return torch.tanh(wf) * torch.sigmoid(wg)


class ResidualBlock(torch.nn.Module):

    """ Residual block: transforms a layer F(x) to F(x) + x """

    def __init__(self, model):
        super(ResidualBlock, self).__init__()

    def forward(self, x):
        out = x + self.model(x)
        return out


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



