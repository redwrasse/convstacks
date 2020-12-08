# models.py
"""
    models and model structures
"""
import building_blocks


def wavenet():

    # wavenet parameters, following the paper
    # https://arxiv.org/pdf/1609.03499.pdf
    # 3-5 blocks each of 10 layers

    dilation_rate = 2
    in_channels = out_channels = 256
    intermediate_channels = 256  # this concept doesn't actually
    # exist for wavenet, need to revise
    kernel_length = 1
    block_n_layers = 10

    bl = building_blocks.Block(n_layers=block_n_layers,
                               kernel_length=kernel_length,
                               dilation_rate=dilation_rate,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               intermediate_channels=intermediate_channels)


def build_custom_model():
    pass

