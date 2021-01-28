# test_buildingblocks.py

import torch
import unittest
import wavenetlike.building_blocks as building_blocks


class TestBuildingBlocks(unittest.TestCase):

    def setUp(self):
        self.example_stack = building_blocks.WavenetLike(
            audio_channel_size=10,
            filter_gate_kernel_size=2,
            dilation_rate=2,
            filter_gate_out_channels=2,
            one_conv_residual_out_channels=2,
            one_conv_skip_out_channels=2,
            block_n_layers=3,
            n_blocks=10)

    def test_stack_properties(self):
        model = self.example_stack
        example_input = torch.ones(size=(5, 10, 4))
        example_output = model.forward(example_input)
        assert example_output.shape == example_input.shape, \
            'stack expected to match input shape'


if __name__ == '__main__':
    unittest.main()


