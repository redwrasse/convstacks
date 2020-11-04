# examples.py
"""
Assuming a finite past time dependence and linear dependence
it becomes an auto-regressive model. This can be trained by
convolutional layers but the time series has to be appropriately
lined up at input and output layers.
The general rules (assuming a left-padding).
# 1) shift output left one index (so an index isn't tied to itself)
        # 2) ignore first k - 1 indices
        # 3) ignore last index in both input and output
        # aka output is the range k-1:-1,
        # input is the range k:
Note this means the kernel size is actually one less than what one may expect,
since an index isn't tied to itself.
What it looks like in this example with k = 2. The first k - 1 (= 1 for k = 2 in this case)
 and last index are ignored for calculating the loss.
    (ig)            (ig)
    x1  x2  x3  x4  x5
    *   *   *   *   *
/   | / | / | / | / |
    *   *   *   *   *
    x0  x1  x2  x3  x4
   (ig)            (ig)
Example auto-regressive model:
    AR(2) process x_t = a x_t-1 + b xt-2 + noise
    stationary if a in [-2, 2], b in [-1, 1]
This trained model then allows prediction, outputting the next timestep value x5.
Iterate to generate a sequence of predictions.
"""
import math
import os
import random
import torch
from utils import ar2_process, download_sample_audio, waveform_to_categorical, \
    waveform_to_input
from stack import Stack, train_stack_ar, analyze_stack,\
    Losses, softmax_loss_fn


def example1():
    """
    train an ar2 model on a single layer convolution
    with an mse loss
    """
    stack = Stack(n_layers=1, kernel_length=2, dilation_rate=1)
    analyze_stack(stack)
    a, b = -0.4, 0.5
    x0, x1 = 50, 60
    n_samples = 100
    data = []
    gen = ar2_process(a, b, x0, x1)
    for i in range(n_samples):
        data.append(gen.__next__())

    train_stack_ar(stack, data, loss_type=Losses.mse)


def example2():
    """ a multilayer convolution network on audio data"""

    print("Training example2: multilayer convolution network on audio data ...")

    CHECKPOINT_SAVE_PATH = './checkpoint'
    MODEL_SAVE_PATH = './model'
    PREDICTIONS_DIR = './predictions'
    KERNEL_SIZE = 10
    N_TRAINING_EPOCHS = 10 ** 5
    NUM_PREDICTIONS = 10

    LEARNING_RATE = 1e-2

    KERNEL_LENGTH = 2
    MU_ENCODING_QUANTIZATION = 256

    data_loader = download_sample_audio(cutoff=100)
    stack = Stack(n_layers=5, kernel_length=KERNEL_LENGTH, dilation_rate=2,
                  n_channels=MU_ENCODING_QUANTIZATION)
    model = stack.model

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE)

    saved_epoch = 0
    if os.path.exists(CHECKPOINT_SAVE_PATH):
        checkpoint = torch.load(CHECKPOINT_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        # saved_epoch_loss = checkpoint['loss']
        print(f"loaded checkpoint model at epoch {saved_epoch}")

    model.train()
    print("training ...")
    # todo("batch training and randomize chunks instead of every chunk")
    for epoch in range(N_TRAINING_EPOCHS):
        epoch_loss = 0.
        # break into chunks to spend less computation time on each iteration
        n_samples = 0
        for i, sample in enumerate(data_loader):
            n_samples += 1
            waveform, sample_rate, labels1, labels2, labels3 = sample
            n = waveform.shape[-1]
            training_sample_length = int(KERNEL_SIZE * 1.5)
            j = random.choice(
                range(0, n - training_sample_length, training_sample_length))
            # for j in range(0, n - training_sample_length, training_sample_length):  # may not be complete
            #   print(j)
            waveform_chunk = waveform[:, :, j: j + training_sample_length]
            categorical_input = waveform_to_categorical(waveform_chunk, m=MU_ENCODING_QUANTIZATION)
            input = waveform_to_input(waveform_chunk, m=MU_ENCODING_QUANTIZATION)
            optimizer.zero_grad()
            output = model(input)
            loss = softmax_loss_fn(output, categorical_input, k=KERNEL_LENGTH,
                                   show_match_fraction=False)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            #print(f'sample chunk loss: {loss.item()}')
        avg_sample_loss = epoch_loss / n_samples
        avg_sample_accuracy = math.exp(-avg_sample_loss)
        if epoch % 3 == 0:
            print(f'*** epoch: {epoch} epoch loss: {epoch_loss} '
                  f'avg. sample loss: {avg_sample_loss} (~{avg_sample_accuracy*100}% accuracy)')
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch + saved_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, CHECKPOINT_SAVE_PATH
            )
            print(f"saved checkpoint at epoch {epoch + saved_epoch}")
            print(f'saving model to {MODEL_SAVE_PATH}...')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    example2()

