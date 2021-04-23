import torch
from torch.nn import functional as F


class LpConv(torch.nn.Conv1d):
    """
    A left-padded convolution
    
    
    Ref. pytorch docs
     output length =
        # (input length + 2 * padding - dilation * (kernel_size - 1) -1 ) / stride + 1
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):  # no convolution biases in wavenet
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
       
    def forward(self, x):
        input_length = x.shape[-1]
        # note self.padding parameter is conv1d double
        #  padding parameter, not our padding
        output_length = \
            int((input_length + 2 * self.padding[0] - self.dilation[0] * 
                 (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        # set additional padding to ensure
        # output length = input length
        __padding = input_length - output_length
        lp_x = F.pad(x,
                     pad=[__padding, 0],
                     mode='constant',
                     value=0)
        return super(LpConv, self).forward(lp_x)
