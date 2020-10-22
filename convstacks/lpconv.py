# lpconv.py
"""
left-padded conv
"""
import torch
import torch.nn.functional as F


class LpConv(torch.nn.Conv1d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(LpConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # verify this relation
        self.__padding = stride * dilation * (kernel_size - 1)

    def forward(self, x):
        lp_x = F.pad(x,
                     pad=self.__padding,
                     mode='constant',
                     value=0)
        return super(LpConv, self).forward(lp_x)



