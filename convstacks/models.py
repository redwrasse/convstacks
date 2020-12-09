# models.py
"""
    models and model structures
"""
import building_blocks
import constants
import torch


def build_wavenet_like(audio_channel_size,
                       filter_gate_kernel_size,
                       dilation_rate,
                       filter_gate_out_channels,
                       one_conv_residual_out_channels,
                       one_conv_skip_out_channels,
                       block_n_layers,
                       n_blocks):

    class WavenetLike(torch.nn.Module):

        def __init__(self):
            super(WavenetLike, self).__init__()

            self.receptive_field_size = sum(dilation_rate**i for i in range(block_n_layers)) * n_blocks

            # add a pre 1-1 conv to change input discretization (from 256) down/up to
            # intermediate channel dimension

            self.pre_one_conv = building_blocks.OneConv(
                in_channels=audio_channel_size,
                out_channels=one_conv_residual_out_channels
            )

            self.layers = torch.nn.ModuleList(
                [WavenetLikeLayer(
                    in_channels=one_conv_residual_out_channels,
                    dilation=dilation_rate ** (
                                i % block_n_layers)
                ) for i in range(block_n_layers *
                                 n_blocks)]
            )
            self.skip_pp = building_blocks.SkipPostProcessing(
                in_channels=one_conv_skip_out_channels,
                intermediate_channels=one_conv_skip_out_channels,
                out_channels=audio_channel_size)

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

        def __init__(self, in_channels, dilation):
            super(WavenetLikeLayer, self).__init__()
            self.gau = building_blocks.GatedActivationUnit()
            self.filter_conv = building_blocks.LpConv(
                in_channels=in_channels,
                out_channels=filter_gate_out_channels,
                kernel_size=filter_gate_kernel_size,
                dilation=dilation
            )
            self.gate_conv = building_blocks.LpConv(
                in_channels=in_channels,
                out_channels=filter_gate_out_channels,
                kernel_size=filter_gate_kernel_size,
                dilation=dilation
            )
            self.residual_one_conv = building_blocks.OneConv(
                in_channels=filter_gate_out_channels,
                out_channels=one_conv_residual_out_channels,
                bias=True
            )
            self.skip_one_conv = building_blocks.OneConv(
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

    return WavenetLike()


def build_wavenet():
    return build_wavenet_like(
        audio_channel_size=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE,
        filter_gate_kernel_size=constants.WaveNetConstants.FILTER_GATE_KERNEL_SIZE,
        dilation_rate=2,
        filter_gate_out_channels=constants.WaveNetConstants.FILTER_GATE_OUT_CHANNELS,
        one_conv_residual_out_channels=constants.WaveNetConstants.ONE_CONV_RESIDUAL_OUT_CHANNELS,
        one_conv_skip_out_channels=constants.WaveNetConstants.ONE_CONV_SKIP_OUT_CHANNELS,
        block_n_layers=constants.WaveNetConstants.BLOCK_N_LAYERS,
        n_blocks=constants.WaveNetConstants.N_BLOCKS
    )

# class Wavenet(torch.nn.Module):
#
#     def __init__(self):
#         super(Wavenet, self).__init__()
#
#         self.receptive_field_size = 1024 * constants.WaveNetConstants.N_BLOCKS # verify
#
#         # add a pre 1-1 conv to change input discretization (from 256) down/up to
#         # intermediate channel dimension
#
#         self.pre_one_conv = building_blocks.OneConv(
#             in_channels=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE,
#             out_channels=constants.WaveNetConstants.ONE_CONV_RESIDUAL_OUT_CHANNELS
#         )
#
#         self.layers = torch.nn.ModuleList(
#             [WavenetLayer(
#                 in_channels=constants.WaveNetConstants.ONE_CONV_RESIDUAL_OUT_CHANNELS,
#                 dilation=2**(i % constants.WaveNetConstants.BLOCK_N_LAYERS)
#             ) for i in range(constants.WaveNetConstants.BLOCK_N_LAYERS *
#                              constants.WaveNetConstants.N_BLOCKS)]
#         )
#         self.skip_pp = building_blocks.SkipPostProcessing(
#             in_channels=constants.WaveNetConstants.ONE_CONV_SKIP_OUT_CHANNELS,
#             intermediate_channels=constants.WaveNetConstants.ONE_CONV_SKIP_OUT_CHANNELS,
#             out_channels=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE)
#
#     def forward(self, x):
#         pre_x = self.pre_one_conv(x)
#         skip_outs = []
#         dense_outs = [pre_x]
#         for i, layer in enumerate(self.layers):
#             out = layer(dense_outs[i])
#             dense_outs.append(out[0])
#             skip_outs.append(out[1])
#         skip_sum = torch.sum(torch.stack(skip_outs), dim=0)
#         out = self.skip_pp(skip_sum)
#         return out
#
#
# class WavenetLayer(torch.nn.Module):
#
#     def __init__(self, in_channels, dilation):
#         super(WavenetLayer, self).__init__()
#         self.gau = building_blocks.GatedActivationUnit()
#         self.filter_conv = building_blocks.LpConv(
#             in_channels=in_channels,
#             out_channels=constants.WaveNetConstants.FILTER_GATE_OUT_CHANNELS,
#             kernel_size=constants.WaveNetConstants.FILTER_GATE_KERNEL_SIZE,
#             dilation=dilation
#         )
#         self.gate_conv = building_blocks.LpConv(
#             in_channels=in_channels,
#             out_channels=constants.WaveNetConstants.FILTER_GATE_OUT_CHANNELS,
#             kernel_size=constants.WaveNetConstants.FILTER_GATE_KERNEL_SIZE,
#             dilation=dilation
#         )
#         self.residual_one_conv = building_blocks.OneConv(
#             in_channels=constants.WaveNetConstants.FILTER_GATE_OUT_CHANNELS,
#             out_channels=constants.WaveNetConstants.ONE_CONV_RESIDUAL_OUT_CHANNELS,
#             bias=True
#         )
#         self.skip_one_conv = building_blocks.OneConv(
#             in_channels=constants.WaveNetConstants.FILTER_GATE_OUT_CHANNELS,
#             out_channels=constants.WaveNetConstants.ONE_CONV_SKIP_OUT_CHANNELS,
#             bias=True
#         )
#
#     def forward(self, x):
#         wf = self.filter_conv(x)
#         wg = self.gate_conv(x)
#         z = self.gau(wf, wg)
#         residual_one_conv_out = self.residual_one_conv(z)
#         skip_out = self.skip_one_conv(z)
#         dense_out = residual_one_conv_out + x
#         return dense_out, skip_out


def build_custom_model():
    pass

