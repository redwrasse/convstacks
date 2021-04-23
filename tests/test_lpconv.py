import unittest
from wavenetlike.lpconv import LpConv
import torch


class TestLpConv(unittest.TestCase):

    def setUp(self):
        self.lpc = LpConv(
            in_channels=1,
            out_channels=1,
            kernel_size=2
        )
        self.x = torch.ones(size=(5, 1, 4))

    def test_forward(self):
        y = self.lpc(self.x)
        assert y.shape == self.x.shape,\
            "left-padded conv should match input shape"


if __name__ == "__main__":
    unittest.main()
