import random
from enum import Enum

import torch
import torchaudio
from torch import __init__


def partial_derivative(y, x, i, j):
    """
    computes del y_i del x_j: partial derivative of y_i wrt x_j

    torch.autograd.grad gives z_i grad_j y_i, where z_i is a vector fed to 'grad_outputs'
    hence feeding a one-hot as z_i gives the jacobian row grad_j y_i for i fixed
    """

    z = torch.zeros(y.shape)
    z[:, :, i] = 1.
    dy_dx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=z,
                                retain_graph=True)[0][:, :, j][0]
    return dy_dx[0].item()


def ar2_process(a, b, x0, x1):
    """
    AR(2) process x_t = a x_t-1 + b xt-2 + noise
    stationary if a in [-2, 2], b in [-1, 1]

    for generating sample data

    a, b parameters
    x0, x1 first two sequence values
    """

    x2 = b * x0 + a * x1
    while True:
        x0 = x1
        x1 = x2
        x2 = b * x0 + a * x1 + random.gauss(0, 10 ** -5)
        yield x2


def download_sample_audio(cutoff=None):
    """65,000 one-second long utterances of 30 short words,
     by thousands of different people
     ref: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
     """
    print('downloading sample audio dataset ....')
    speech_commands = torchaudio.datasets.SPEECHCOMMANDS('.', download=True)
    data_loader = torch.utils.data.DataLoader(speech_commands,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)
    print('finished downloading sample audio dataset.')
    if cutoff is not None:
        mini_data_loader = []
        for i, e in enumerate(data_loader):
            mini_data_loader.append(e)
            if i > cutoff:
                return mini_data_loader
    else:
        return data_loader


def __get_encoding(m):
    # to do remove dep. on torchaudio, implement mu encoding directly
    return torchaudio.transforms.MuLawEncoding(quantization_channels=m)


def __get_decoding(m):
    # to do remove dep. on torchaudio, implement mu encoding/decoding directly
    return torchaudio.transforms.MuLawDecoding(quantization_channels=m)


def waveform_to_categorical(waveform, m):

    assert torch.max(waveform) <= 1. and torch.min(waveform) >= -1., \
        "mu encoding input not within required bounds [-1, 1]"
    enc = __get_encoding(m)
    return enc(waveform)


def waveform_to_input(waveform, m):
    assert waveform.shape[1] == 1, "expected single channel waveform"
    categorical = waveform_to_categorical(waveform, m)
    z = torch.nn.functional.one_hot(categorical, m)
    #!! for the time being assume single channel so can squeeze dim=1 (can generalize later)
    return torch.squeeze(torch.nn.functional.one_hot(categorical, m),
                    dim=1).permute(0, 2, 1).float()


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
    # k should be model receptive field size
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
