"""

wavenet constants

Usage (with the architecture as in the DeepMind paper):


tbd: verify, from both paper and ibab's implementation


        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # kernel size
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)

ref: https://github.com/ibab/tensorflow-wavenet/pull/229
and https://github.com/ibab/tensorflow-wavenet/issues/227


discussion there about selecting various parameters, also including
discriminiatve gaussian as possible training in addition to discriminative
categorical

(lots of other discussion in other issues about changing model architecture)


wavenet parameters in ibab's implementation

https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet_params.json
{
    "filter_width": 2,
    "sample_rate": 16000,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "residual_channels": 32,
    "dilation_channels": 32,
    "quantization_channels": 256,
    "skip_channels": 512,
    "use_biases": true,
    "scalar_input": false,
    "initial_filter_width": 32
}
"""


class WaveNetConstants:

    # mu-encoding discretization
    AUDIO_CHANNEL_SIZE = 256
    MU = AUDIO_CHANNEL_SIZE - 1

    # kernel size of filter and gate convolutions. Should be 2, as from the
    # paper
    FILTER_GATE_KERNEL_SIZE = 2

    # number of output channels in filter and gate convolutions
    FILTER_GATE_OUT_CHANNELS = 32
    # number of output channels in 1 by 1 conv for residual,
    # aka dense output. This means all intermediate layer dense inputs/convolutions
    # operate on this same reduced channel size/discretization?
    ONE_CONV_RESIDUAL_OUT_CHANNELS = 32
    #  number of output channels in 1 by 1 conv
    #  for skip output
    ONE_CONV_SKIP_OUT_CHANNELS = 512

    # number of layers per block
    BLOCK_N_LAYERS = 10

    # Inferred from 200-300ms receptive field size in
    # Wavenet paper ~ 3-5 blocks.
    # Selection depends on desired receptive field size
    N_BLOCKS = 3

    # length of timeseries chunks to feed to wavenet
    # should just be the total model
    # receptive field length rather than an independent constant?
    INPUT_CHUNK_LENGTH = 1000

