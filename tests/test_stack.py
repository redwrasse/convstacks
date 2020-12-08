# test_stack.py
import torch
import unittest
from convstacks.building_blocks import Block
from ops import mse_loss_fn
from train import train_stack_ar
from analyzer import analyze_stack


class TestStack(unittest.TestCase):

    def setUp(self):
        self.example_stack = Block(
            n_layers=5,
            kernel_length=2,
            dilation_rate=2
        )

    def test_stack_properties(self):
        model = self.example_stack.model
        example_input = torch.ones(size=(5, 1, 4))
        example_output = model.forward(example_input)
        assert example_output.shape == example_input.shape, \
            'stack expected to match input shape'

    def test_analyze_stack(self):
        analyze_stack(self.example_stack)

    def test_mse_loss_fn(self):
        # should have zero loss for output y such that y_i = x_i+1
        x = torch.randn(size=(10, 4, 6))
        y = x[:, :, 1:]
        x2 = x[:, :, :-1]
        loss = mse_loss_fn(y, x2, k=2)
        assert torch.eq(loss, torch.Tensor([0.,])),\
            "expected zero loss on mse loss function"

    def test_softmax_loss_fn(self):
        pass

    def test_train_stack_ar(self):
        pass


if __name__ == '__main__':
    unittest.main()


