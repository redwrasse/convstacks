# test_stack.py
import torch
import unittest
from convstacks.stack import Stack,\
    analyze_stack, train_stack_ar
from utils import ar2_process


class TestStack(unittest.TestCase):

    def setUp(self):
        self.example_stack = Stack(
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

    def test_train_stack_ar(self):
        pass


if __name__ == '__main__':
    unittest.main()


