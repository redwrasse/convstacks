# wavenet_example.py

from models import build_wavenet,\
  build_wavenet_toy
import train
import dataset


def ar2_example():
    pass


def wavenet_example():

    model = build_wavenet()
    data = dataset.SpeechCommands(cutoff=100)
    train.train(model, data)


def wavenet_simple_example():

    model = build_wavenet_toy()
    data = dataset.SpeechCommands(cutoff=5)
    train.train(model, data)


if __name__ == "__main__":
    wavenet_simple_example()


