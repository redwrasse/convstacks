import unittest
import torch
from utils import partial_derivative, ar2_process, \
    waveform_to_input, waveform_to_categorical


class TestUtils(unittest.TestCase):

    def test_partial_derivative(self):
        pass

    def test_get_decoding(self):
        pass

    def test_waveform_to_categorical(self):
        example_input = torch.clamp(torch.randn(size=(5, 1, 4)),
                                    min=-1., max=1.)
        example_output = waveform_to_categorical(example_input, m=10)

        assert example_output.shape == example_input.shape
        # check all values in set {0, 1, ..., 9}
        encodings = set(range(10))
        for row in example_output:
            for channel in row:
                for e in channel:
                    el = int(e.data.cpu().numpy())
                    assert el in encodings

    def test_waveform_to_input(self):
        example_input = torch.clamp(torch.randn(size=(5, 1, 4)),
                                    min=-1., max=1.)
        m = 10
        example_output = waveform_to_input(example_input, m=m)
        assert example_output.shape == (example_input.shape[0], m,
                                        example_input.shape[2]),\
            "expected one hot encoded input"

    def test_ar2_process(self):
        pass


if __name__ == "__main__":
    unittest.main()

