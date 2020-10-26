# stack.py
from lpconv import LpConv
import torch


class Stack:

    def __init__(self, n_layers, kernel_length, dilation_rate):
        self.n_layers = n_layers
        self.kernel_length = kernel_length
        self.dilation_rate = dilation_rate
        self.model = _build_stack(n_layers=n_layers,
                                  kernel_length=kernel_length,
                                  dilation_rate=dilation_rate)


def _build_stack(n_layers, kernel_length, dilation_rate=2):
    # to do: generalize parameters
    # effective kernel length with stride of 1 for n layers is
    # sum_{i=0 to n-1} kernel_length * dilation_rate**i
    lpc = LpConv(in_channels=1, out_channels=1, kernel_size=kernel_length,
                 dilation=dilation_rate**0)
    for i in range(n_layers-1):
        lpc = torch.nn.Sequential(lpc, LpConv(in_channels=1,
                                              out_channels=1,
                                              kernel_size=kernel_length,
                                              dilation=dilation_rate**(i+1)))
    return lpc


def analyze_stack(stack):
    # ought to determine effective kernel length
    # for loss function
    # use partial derivatives
    effective_kernel_length = sum(stack.kernel_length * stack.dilation_rate**i
                                  for i in range(stack.n_layers))
    print(f'stack with effective kernel length of {effective_kernel_length}')


def train_stack(stack, data):

    def loss_fn(output, input, k):
        modified_out = output[:, :, k - 1:-1]
        modified_in = input[:, :, k:]
        return torch.nn.MSELoss()(modified_out,
                                  modified_in)

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
