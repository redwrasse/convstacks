import torch


class ResidualBlock(torch.nn.Module):

    """ Residual block: transforms a layer F(x) to F(x) + x """

    def __init__(self, model):
        super(ResidualBlock, self).__init__()
        self.model = model

    def forward(self, x):
        out = x + self.model(x)
        return out
