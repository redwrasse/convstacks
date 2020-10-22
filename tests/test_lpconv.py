import unittest
from convstacks.lpconv import LpConv


class TestLpConv(unittest.TestCase):

    def setUp(self):
        self.lpc = LpConv(
            in_channels=1,
            out_channels=1,
            kernel_size=3
        )

    def test_forward(self):
        pass


if __name__ == "__main__":
    unittest.main()
