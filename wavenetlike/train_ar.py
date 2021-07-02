import torch
from torch.utils.tensorboard import SummaryWriter

import ops as ops
from train import logger


def train_stack_ar(model, dataset):
    # train an autoregressive model on the data, using the stack

    writer = SummaryWriter()

    def make_batched(data, batch_size):
        batched = []
        batch = []
        for e in data:
            if len(batch) == batch_size:
                batched.append(batch)
                batch = []
            batch.append(e)
        return torch.FloatTensor(batched).unsqueeze(1)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3)
    n_epochs = 10 ** 3
    batch_size = 5  # should actually set to some multiple of the
    # effective kernel length
    batched = make_batched(dataset, batch_size)
    n_batches = batched.shape[1]
    k = 2  # effective kernel length
    loss_fn = ops.mse_loss_fn

    # if loss_type == ops.Losses.mse:
    #     logger.info("Training with mse loss")
    #     loss_fn = ops.mse_loss_fn
    # else:
    #     logger.info("Training with softmax loss")
    #     loss_fn = ops.softmax_loss_fn

    for epoch in range(n_epochs):
        epoch_loss = 0.
        for j in range(n_batches):
            optimizer.zero_grad()
            chunk = batched[j*batch_size:(j + 1) * batch_size, :, :]
            y = model(chunk)
            loss = loss_fn(y, chunk, k)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        if epoch % 100 == 0:
            logger.info(f'epoch loss: {epoch_loss}')
        # log epoch loss to tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)

