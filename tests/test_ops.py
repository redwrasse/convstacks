import unittest
import torch
from wavenetlike.ops import partial_derivative, ar2_process, \
    waveform_to_input, waveform_to_categorical, \
    mse_loss_fn


class TestOps(unittest.TestCase):

    def setUp(self):
        self.sample_input = torch.clamp(torch.randn(size=(5, 1, 4)),
                                    min=-1., max=1.)

    def test_partial_derivative(self):
        # input length and kernel size
        n, k = 10, 3
        x = torch.randn(1, 1, n)
        x.requires_grad = True
        layer = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k
        )
        layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
        y = layer(x)
        assert y.shape[2] == 8, \
            'standard convolution output should be of length n - k + 1'
        for i in range(n - k + 1):
            for j in range(n):
                # nonzero values should be for which w_{i + k - j -1}
                # is nonzero, eg i + k - j - 1 = 0, ..., k - 1 (see notes at top)
                pd = partial_derivative(y, x, i, j)
                in_range = (i - j + k - 1 >= 0) and (i - j + k - 1 <= k - 1)
                if in_range:
                    assert pd == 1.0, \
                        'unexpected standard convolution partial derivative'
                else:
                    assert pd == 0.0, \
                        'unexpected standard convolution partial derivative'

    def test_waveform_to_categorical(self):
        example_output = waveform_to_categorical(self.sample_input, m=10)
        # check shape
        assert example_output.shape == self.sample_input.shape
        # check all values in set {0, 1, ..., 9}
        encodings = set(range(10))
        for row in example_output:
            for channel in row:
                for e in channel:
                    el = int(e.data.cpu().numpy())
                    assert el in encodings

    def test_waveform_to_input(self):
        m = 10
        example_output = waveform_to_input(self.sample_input, m=m)
        assert example_output.shape == (self.sample_input.shape[0], m,
                                        self.sample_input.shape[2]),\
            "expected one hot encoded input"

    def test_ar2_process(self):
        pass

    def test_mse_loss_fn(self):
        # should have zero loss for output y such that y_i = x_i+1
        x = torch.randn(size=(10, 4, 6))
        y = x[:, :, 1:]
        x2 = x[:, :, :-1]
        loss = mse_loss_fn(y, x2, k=2)
        assert torch.eq(loss, torch.Tensor([0., ])), \
            "expected zero loss on mse loss function"


if __name__ == "__main__":
    unittest.main()

