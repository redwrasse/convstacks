import unittest
from convstacks.lpconv import LpConv
import torch


class TestLpConv(unittest.TestCase):

    def setUp(self):
        self.lpc = LpConv(
            in_channels=1,
            out_channels=1,
            kernel_size=2
        )

    def test_forward(self):
        x = torch.ones(size=(5, 1, 4))
        y = self.lpc(x)
        assert y.shape == x.shape, "should match input shape"


if __name__ == "__main__":
    unittest.main()
