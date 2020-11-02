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


def mu_encoding(x, quantization_channels):
    # to do: remove dependence on torchaudio, implement
    # mu encoding directly
    assert torch.max(x) <= 1. and torch.min(x) >= -1., \
        "mu encoding input not within required bounds [-1, 1]"
    return torchaudio.functional.mu_law_encoding(x,
                                                 quantization_channels=quantization_channels)

# for audio_data in YESNO_DATA:
#      waveform, sample_rate, labels = audio_data
#
#      quantized_waveform = torch.nn.functional.one_hot(torchaudio.functional.mu_law_encoding(waveform,
#                                            quantization_channels=256), 256).permute(0, 2, 1)
#      n = quantized_waveform.shape[2]
#      indices = list(range(int(n/INPUT_CHUNK_LENGTH)))
#      random.shuffle(indices)
#      for i in indices:
#          chunk = quantized_waveform[:, :,
#                  i * INPUT_CHUNK_LENGTH: (i+1) * INPUT_CHUNK_LENGTH].float()
#          yield chunk
