# train.py
"""
    training loops
"""
import torch
import os

import ops


class WavenetStepOp:

    def __init__(self, audio_channel_size, receptive_field_size):
        self.audio_channel_size = audio_channel_size
        self.receptive_field_size = receptive_field_size

    def step(self, optimizer, model, audio_sample):
        waveform, sample_rate, labels1, labels2, labels3 = audio_sample
        x = ops.waveform_to_input(waveform,
                                  m=self.audio_channel_size)
        cx = ops.waveform_to_categorical(waveform,
                                         m=self.audio_channel_size)
        # print(f'input shape: {x.shape}')
        y = model(x)
        # print(f'output shape: {y.shape}')

        optimizer.zero_grad()
        loss = ops.softmax_loss_fn(y, cx, self.receptive_field_size,
                                   show_match_fraction=True)
        loss.backward()
        optimizer.step()
        return loss.item()


def train(model,
          dataset,
          learning_rate=1e-3,
          nepochs=10**5):

    checkpt_path = "./checkpoint"
    model_save_path = "./model"

    learning_rate = learning_rate
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    stepOp = WavenetStepOp(model.in_out_channel_size, model.receptive_field_size)

    saved_epoch = 0

    print("checking for existing checkpointed model ...")
    if os.path.exists(checkpt_path):
        checkpoint = torch.load(checkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        # saved_epoch_loss = checkpoint['loss']
        print(f"loaded checkpoint model at epoch {saved_epoch}")
    else:
        print("no checkpointed model found.")

    model.train()
    print("training ...")
    for epoch in range(nepochs):
        for i, audio_sample in enumerate(dataset):
            loss_value = stepOp.step(optimizer, model, audio_sample)

            print(f'(epoch = {epoch}) loss: {loss_value}')

            if epoch > 10**5:
                break
            if epoch % 2 == 0:
                torch.save({
                    'epoch': epoch + saved_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                }, checkpt_path
                )
                print(f"saved checkpoint at epoch {epoch + saved_epoch}")
                print(f'saving model to {model_save_path}...')
                torch.save(model.state_dict(), model_save_path)


def train_stack_ar(model, data, loss_type):

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

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3)
    n_epochs = 10**3
    batch_size = 5  # should actually set to some multiple of the
    # effective kernel length
    batched = make_batched(data, batch_size)
    n_batches = batched.shape[1]
    k = 2  # effective kernel length

    if loss_type == ops.Losses.mse:
        loss_fn = ops.mse_loss_fn
    else:
        loss_fn = ops.softmax_loss_fn

    for epoch in range(n_epochs):
        epoch_loss = 0.
        for j in range(n_batches):
            optimizer.zero_grad()
            chunk = batched[j*batch_size:(j+1)*batch_size, :, :]
            y = model(chunk)
            loss = loss_fn(y, chunk, k)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'epoch loss: {epoch_loss}')

