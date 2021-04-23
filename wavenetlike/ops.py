import random
from enum import Enum
import logging

import torch
import torchaudio


logger = logging.getLogger(__name__)


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
        x0, x1, x2 = x1, x2, b * x0 + a * x1 + random.gauss(0, 10 ** -5)
        yield x2


def mu_law_encoding(
        x,
        quantization_channels):
    r"""Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1.

    Args:
        x (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law encoding
    """
    mu = quantization_channels - 1.0
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mu_law_decoding(
        x_mu,
        quantization_channels):
    r"""Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law decoding
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = ((x_mu) / mu) * 2 - 1.0
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


def waveform_to_categorical(waveform, m):

    assert torch.max(waveform) <= 1. and torch.min(waveform) >= -1., \
        "mu encoding input not within required bounds [-1, 1]"
    enc = mu_law_encoding(waveform, m)
    return enc


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
    assert loss_output.shape[1:] == (M, N)
    loss_input = torch.squeeze(input[:, :, k:],
                               dim=1)
    assert loss_input.shape[1:] == (N,)
    loss_fn = torch.nn.CrossEntropyLoss()
    if show_match_fraction:
        fraction_matched = __match_fraction_internal(loss_output, loss_input)
        logger.debug(f'matched for loss on '
              f'{fraction_matched * 100}% of quantized encoding values')
    return loss_fn(loss_output, loss_input)


def match_fraction(output, input, k):
    loss_output = output[:, :, k - 1:-1]
    loss_input = torch.squeeze(input[:, :, k:],
                               dim=1)
    fraction_matched = __match_fraction_internal(loss_output, loss_input)
    return fraction_matched


def __match_fraction_internal(loss_output, loss_input):
    loss_output_categorical = logit_to_categorical(loss_output)  # for debugging
    if_true = torch.sum((loss_output_categorical == loss_input).float())
    if_false = torch.sum((loss_output_categorical != loss_input).float())
    fraction_matched = if_true / (if_false + if_true)
    return fraction_matched


def logit_to_categorical(logit):
    return torch.argmax(torch.softmax(logit, dim=1), dim=1, keepdim=True)
