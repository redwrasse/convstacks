# models.py
"""
    models and model structures
"""
import wavenetlike.lpconv as lpconv
import wavenetlike.constants as constants
from wavenetlike.wavenetlike import WavenetLike


def build_wavenet_like(audio_channel_size,
                       filter_gate_kernel_size,
                       dilation_rate,
                       filter_gate_out_channels,
                       one_conv_residual_out_channels,
                       one_conv_skip_out_channels,
                       block_n_layers,
                       n_blocks):

    return WavenetLike(
        audio_channel_size=audio_channel_size,
        filter_gate_kernel_size=filter_gate_kernel_size,
        dilation_rate=dilation_rate,
        filter_gate_out_channels=filter_gate_out_channels,
        one_conv_residual_out_channels=one_conv_residual_out_channels,
        one_conv_skip_out_channels=one_conv_skip_out_channels,
        block_n_layers=block_n_layers,
        n_blocks=n_blocks

    )


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


def build_wavenet_toy():
    return build_wavenet_like(
        audio_channel_size=constants.WaveNetToyConstants.AUDIO_CHANNEL_SIZE,
        filter_gate_kernel_size=constants.WaveNetToyConstants.FILTER_GATE_KERNEL_SIZE,
        dilation_rate=2,
        filter_gate_out_channels=constants.WaveNetToyConstants.FILTER_GATE_OUT_CHANNELS,
        one_conv_residual_out_channels=constants.WaveNetToyConstants.ONE_CONV_RESIDUAL_OUT_CHANNELS,
        one_conv_skip_out_channels=constants.WaveNetToyConstants.ONE_CONV_SKIP_OUT_CHANNELS,
        block_n_layers=constants.WaveNetToyConstants.BLOCK_N_LAYERS,
        n_blocks=constants.WaveNetToyConstants.N_BLOCKS
    )


def build_ar2():
    model = lpconv.LpConv(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        dilation=1
    )
    return model


def build_custom_model():
    pass

