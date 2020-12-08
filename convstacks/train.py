# train.py
"""
    training loops
"""
import torch
from ops import Losses, mse_loss_fn, softmax_loss_fn


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

