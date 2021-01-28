import unittest
import torch
from wavenetlike.ops import partial_derivative, ar2_process, \
    waveform_to_input, waveform_to_categorical, \
    download_sample_audio, mse_loss_fn


class TestOps(unittest.TestCase):

    def test_partial_derivative(self):
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

    def test_download_sample_audio(self):
        download_sample_audio()

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

