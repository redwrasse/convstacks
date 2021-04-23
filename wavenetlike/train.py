# train.py
"""
    training loops
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging

import wavenetlike.ops as ops


logger = logging.getLogger(__name__)


class WavenetStepOp:

    def __init__(self, audio_channel_size, receptive_field_size):
        self.audio_channel_size = audio_channel_size
        self.receptive_field_size = receptive_field_size

    def step(self, optimizer, model, audio_sample, epoch, writer=None):
        waveform, sample_rate, labels1, labels2, labels3 = audio_sample
        x = ops.waveform_to_input(waveform,
                                  m=self.audio_channel_size)
        cx = ops.waveform_to_categorical(waveform,
                                         m=self.audio_channel_size)
        # logger.info(f'input shape: {x.shape}')
        y = model(x)
        # logger.info(f'output shape: {y.shape}')

        optimizer.zero_grad()
        loss = ops.softmax_loss_fn(y, cx, self.receptive_field_size,
                                   show_match_fraction=True)
        fmatched = ops.match_fraction(y, cx, self.receptive_field_size)
        if writer:
            writer.add_scalar("Fraction matched/train", fmatched, epoch)
        logger.info(f'fraction matched: {fmatched * 100}%')
        loss.backward()
        optimizer.step()
        return loss.item()


def train(model,
          dataset,
          learning_rate=1e-3,
          nepochs=10**5,
          epoch_save_freq=2):

    writer = SummaryWriter()

    train_artifacts_dir = os.path.join(os.curdir, 'trainartifacts')
    if not os.path.exists(train_artifacts_dir):
        os.mkdir(train_artifacts_dir)

    checkpt_path = os.path.join(train_artifacts_dir, "checkpoint")
    model_save_path = os.path.join(train_artifacts_dir, "model")

    learning_rate = learning_rate
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    stepOp = WavenetStepOp(model.in_out_channel_size, model.receptive_field_size)

    saved_epoch = 0

    logger.info("checking for existing checkpointed model ...")
    if os.path.exists(checkpt_path):
        checkpoint = torch.load(checkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        # saved_epoch_loss = checkpoint['loss']
        logger.info(f"loaded checkpoint model at epoch {saved_epoch}")
    else:
        logger.info("no checkpointed model found.")

    model.train()
    logger.info("training ...")

    for epoch in range(nepochs):
        epoch_loss = 0.
        n_samples = 0.
        for i, audio_sample in enumerate(dataset):
            loss_value = stepOp.step(optimizer, model, audio_sample, epoch,
                                     writer)
            epoch_loss += loss_value
            n_samples += 1
        epoch_loss /= n_samples
        logger.info(f'(epoch = {epoch}) epoch loss: {epoch_loss}')

        writer.add_scalar("Loss/train", epoch_loss, epoch)

        if epoch > nepochs:
            break
        if epoch % epoch_save_freq == 0:
            torch.save({
                'epoch': epoch + saved_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, checkpt_path
            )
            logger.info(f"saved checkpoint at epoch {epoch + saved_epoch}")
            logger.info(f'saving model to {model_save_path}...')
            torch.save(model.state_dict(), model_save_path)


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

