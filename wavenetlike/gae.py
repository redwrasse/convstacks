import torch


class GatedActivationUnit(torch.nn.Module):

    """ Gated activation unit from wavenet """

    def __init__(self):
        super(GatedActivationUnit, self).__init__()

    def forward(self, wf, wg):
        return torch.tanh(wf) * torch.sigmoid(wg)

