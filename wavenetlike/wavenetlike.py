import logging
import torch

from wavenetlike.gae import GatedActivationUnit
from wavenetlike.lpconv import LpConv
from wavenetlike.oneconv import OneConv
from wavenetlike.skippp import SkipPostProcessing

logger = logging.getLogger(__name__)


class WavenetLike(torch.nn.Module):

    def __init__(self,
                 audio_channel_size,
                 dilation_rate,
                 filter_gate_kernel_size,
                 filter_gate_out_channels,
                 one_conv_residual_out_channels,
                 one_conv_skip_out_channels,
                 block_n_layers,
                 n_blocks):
        super(WavenetLike, self).__init__()

        self.receptive_field_size = sum(dilation_rate**i for i in range(block_n_layers)) * n_blocks
        self.in_out_channel_size = audio_channel_size

        # add a pre 1-1 conv to change input discretization
        # (from 256 down/up to intermediate channel dimension)
        self.pre_one_conv = OneConv(
            in_channels=audio_channel_size,
            out_channels=one_conv_residual_out_channels
        )

        self.layers = torch.nn.ModuleList(
            [WavenetLikeLayer(
                in_channels=one_conv_residual_out_channels,
                dilation=dilation_rate ** (
                            i % block_n_layers),
                filter_gate_out_channels=filter_gate_out_channels,
                filter_gate_kernel_size=filter_gate_kernel_size,
                one_conv_residual_out_channels=one_conv_residual_out_channels,
                one_conv_skip_out_channels=one_conv_skip_out_channels
            ) for i in range(block_n_layers *
                             n_blocks)]
        )
        self.skip_pp = SkipPostProcessing(
            in_channels=one_conv_skip_out_channels,
            intermediate_channels=one_conv_skip_out_channels,
            out_channels=audio_channel_size)

        logger.info(f'Built wavenetlike model with receptive field size of {self.receptive_field_size}')


    def forward(self, x):
        pre_x = self.pre_one_conv(x)
        skip_outs = []
        dense_outs = [pre_x]
        for i, layer in enumerate(self.layers):
            out = layer(dense_outs[i])
            dense_outs.append(out[0])
            skip_outs.append(out[1])
        skip_sum = torch.sum(torch.stack(skip_outs), dim=0)
        out = self.skip_pp(skip_sum)
        return out


class WavenetLikeLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 dilation,
                 filter_gate_out_channels,
                 filter_gate_kernel_size,
                 one_conv_residual_out_channels,
                 one_conv_skip_out_channels):
        super(WavenetLikeLayer, self).__init__()
        self.gau = GatedActivationUnit()
        self.filter_conv = LpConv(
            in_channels=in_channels,
            out_channels=filter_gate_out_channels,
            kernel_size=filter_gate_kernel_size,
            dilation=dilation
        )
        self.gate_conv = LpConv(
            in_channels=in_channels,
            out_channels=filter_gate_out_channels,
            kernel_size=filter_gate_kernel_size,
            dilation=dilation
        )
        self.residual_one_conv = OneConv(
            in_channels=filter_gate_out_channels,
            out_channels=one_conv_residual_out_channels,
            bias=True
        )
        self.skip_one_conv = OneConv(
            in_channels=filter_gate_out_channels,
            out_channels=one_conv_skip_out_channels,
            bias=True
        )

    def forward(self, x):
        wf = self.filter_conv(x)
        wg = self.gate_conv(x)
        z = self.gau(wf, wg)
        residual_one_conv_out = self.residual_one_conv(z)
        skip_out = self.skip_one_conv(z)
        dense_out = residual_one_conv_out + x
        return dense_out, skip_out
