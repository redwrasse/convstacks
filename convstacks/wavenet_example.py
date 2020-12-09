# wavenet_example.py

from models import build_wavenet,\
  build_wavenet_toy
import train
import constants
import ops
import os
import torch


def wavenet_example():

    model = build_wavenet()
    dataset = ops.download_sample_audio(cutoff=100)
    train.train(model, dataset)


def wavenet_simple_example():

    model = build_wavenet_toy()
    dataset = ops.download_sample_audio(cutoff=5)
    train.train(model, dataset)


if __name__ == "__main__":
    wavenet_simple_example()


