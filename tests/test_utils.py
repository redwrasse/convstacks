import unittest
import torch
from utils import partial_derivative, ar2_process, \
    mu_encoding


class TestUtils(unittest.TestCase):

    def test_mu_encoding(self):
        example_input = torch.clamp(torch.randn(size=(5, 1, 4)),
                                    min=-1., max=1.)
        example_output = mu_encoding(example_input,
                                     quantization_channels=10)
        assert example_output.shape == example_input.shape
        # check all values in set {0, 1, ..., 9}
        encodings = set(range(10))
        for row in example_output:
            for channel in row:
                for e in channel:
                    el = int(e.data.cpu().numpy())
                    assert el in encodings

    def test_partial_derivative(self):
        pass

    def test_ar2_process(self):
        pass


if __name__ == "__main__":
    unittest.main()

