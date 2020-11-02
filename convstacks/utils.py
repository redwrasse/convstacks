import random
import torch
import torchaudio


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


def get_encoding(m):
    # to do remove dep. on torchaudio, implement mu encoding directly
    return torchaudio.transforms.MuLawEncoding(quantization_channels=m)


def get_decoding(m):
    # to do remove dep. on torchaudio, implement mu encoding/decoding directly
    return torchaudio.transforms.MuLawDecoding(quantization_channels=m)


def waveform_to_categorical(waveform, m):

    assert torch.max(waveform) <= 1. and torch.min(waveform) >= -1., \
        "mu encoding input not within required bounds [-1, 1]"
    enc = get_encoding(m)
    return enc(waveform)


def waveform_to_input(waveform, m):
    assert waveform.shape[1] == 1, "expected single channel waveform"
    categorical = waveform_to_categorical(waveform, m)
    z = torch.nn.functional.one_hot(categorical, m)
    #!! for the time being assume single channel so can squeeze dim=1 (can generalize later)
    return torch.squeeze(torch.nn.functional.one_hot(categorical, m),
                    dim=1).permute(0, 2, 1).float()

