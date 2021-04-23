import unittest
import torch
from wavenetlike.residual import ResidualBlock


class TestResidualBlock(unittest.TestCase):
    
    def setUp(self):
        class FooModel(torch.nn.Module):
            def __init__(self):
                super(FooModel, self).__init__()

            def forward(self, x): return x

        self.residual_block = ResidualBlock(FooModel())
        self.x = torch.ones(size=(5, 1, 4))

    def test_forward(self):
        y = self.residual_block(self.x)
        assert torch.all(torch.eq(y, self.x + 1))



