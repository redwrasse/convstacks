# stack.py
"""


x0, .... x5 (n = 6)
with k = 2

x2 = f(x0, x1)
...
x5 = f(x3, x4)


x0, ..., xn-1 inputs
y0, ....,yn-1 outputs
model as sequence of convolutional layers w/left padding

k effective kernel size:
	xk  = f(x0, ..., xk-1)
	...
	xn-1 = f(xn-1-k, ..., xn-2)


graphically lth output gets tied to lth input and preceding k - 1 inputs

yl = f(xl-k+1,...xl)

y0	y1  y2  y3  ... yn-1
*	*	*	*	*	*

 (convolutional layers)

*	*   *   *	*	*
x0  x1  x2  x3  ... xn-1

So match model by matching output to input as y1 = x2, y2 = x3 etc,
so that xl+1 = f(xl-k+1,...xl), as desired.

so train model with loss function to match yl to xl+1.
in loss function keep only full relationships: ignore first k - 1 outputs,
ignore last output, ignore last input.

loss = sum_i (i = k-1, n-2) el_loss(y_i, x_i+1)

so looks like

	    xk  ...        xn-1
		*	*	*	*	*

 (convolutional layers)

*	*   *   *	*	*	*
x0  .. xk-1  ....      xn-2



"""
from enum import Enum
from lpconv import LpConv
import torch


class Stack:
    # TODO("generalize to arbitrary intermediate channel size")
    def __init__(self, n_layers, kernel_length, dilation_rate=2, n_channels=1):
        self.n_layers = n_layers
        self.kernel_length = kernel_length
        self.dilation_rate = dilation_rate
        self.model = _build_stack(n_layers=n_layers,
                                  kernel_length=kernel_length,
                                  dilation_rate=dilation_rate,
                                  n_channels=n_channels)


def _build_stack(n_layers, kernel_length, dilation_rate, n_channels):
    # to do: generalize parameters
    # effective kernel length with stride of 1 for n layers is
    # sum_{i=0 to n-1} kernel_length * dilation_rate**i
    lpc = LpConv(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_length,
                 dilation=dilation_rate**0)
    for i in range(n_layers-1):
        lpc = torch.nn.Sequential(lpc, LpConv(in_channels=n_channels,
                                              out_channels=n_channels,
                                              kernel_size=kernel_length,
                                              dilation=dilation_rate**(i+1)))
    return lpc


def analyze_stack(stack):
    # ought to determine effective kernel length
    # for loss function
    # the effective kernel length is defined as the longest previous dependence
    # aka if x_n = f(x_(n-k), x_(n-k+1),...) then the effective kernel length is k
    # use partial derivatives
    effective_kernel_length = sum(stack.kernel_length * stack.dilation_rate**i
                                  for i in range(stack.n_layers))
    print(f'stack with effective kernel length of {effective_kernel_length}')


class Losses(Enum):

    mse = 1
    softmax = 2


def mse_loss_fn(output, input, k):
    # k is effective kernel size
    modified_out = output[:, :, k - 1:-1]
    modified_in = input[:, :, k:]
    return torch.nn.MSELoss()(modified_out,
                              modified_in)


def softmax_loss_fn(output, input, k, show_match_fraction=False):
    # input expected of shape (n, 1, l), w/categorical values
    # output expected of shape (n, m, l), where m is number of categories
    # should be logits (aka unnormalized) distribution over all m categories
    # k is effective kernel size
    # per element softmax
    loss_output = output[:, :, k - 1:-1]
    N = loss_output.shape[-1]
    M = loss_output.shape[-2]
    assert loss_output.shape == (1, M, N)
    loss_input = torch.squeeze(input[:, :, k:],
                               dim=1)
    assert loss_input.shape == (1, N)
    loss_fn = torch.nn.CrossEntropyLoss()
    if show_match_fraction:
        fraction_matched = match_fraction(loss_output, loss_input)
        print(f'matched for loss on '
              f'{fraction_matched * 100}% of quantized encoding values')
    return loss_fn(loss_output, loss_input)


def match_fraction(loss_output, loss_input):
    loss_output_categorical = logit_to_categorical(loss_output)  # for debugging
    if_true = torch.sum((loss_output_categorical == loss_input).float())
    if_false = torch.sum((loss_output_categorical != loss_input).float())
    fraction_matched = if_true / (if_false + if_true)
    return fraction_matched


def logit_to_categorical(logit):
    return torch.argmax(torch.softmax(logit, dim=1), dim=1, keepdim=True)


def train_stack_ar(stack, data, loss_type):
    # train an autoregressive model on the data, using the stack

    def make_batched(data, batch_size):
        batched = []
        batch = []
        for e in data:
            if len(batch) == batch_size:
                batched.append(batch)
                batch = []
            batch.append(e)
        return torch.FloatTensor(batched).unsqueeze(1)

    optimizer = torch.optim.SGD(stack.model.parameters(),
                                lr=1e-3)
    n_epochs = 10**3
    batch_size = 5  # should actually set to some multiple of the
    # effective kernel length
    batched = make_batched(data, batch_size)
    n_batches = batched.shape[1]
    k = 2  # effective kernel length

    if loss_type == Losses.mse:
        loss_fn = mse_loss_fn
    else:
        loss_fn = softmax_loss_fn

    for epoch in range(n_epochs):
        epoch_loss = 0.
        for j in range(n_batches):
            optimizer.zero_grad()
            chunk = batched[j*batch_size:(j+1)*batch_size, :, :]
            y = stack.model(chunk)
            loss = loss_fn(y, chunk, k)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'epoch loss: {epoch_loss}')


if __name__ == "__main__":
    pass
