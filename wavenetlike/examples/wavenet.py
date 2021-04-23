import wavenetlike.train as train
from wavenetlike.models import build_wavenet
from wavenetlike.dataset import Dataset


def wavenet_example():
    model = build_wavenet()
    dataset = Dataset(key="SPEECHCOMMANDS",
                      cutoff=100)
    train.train(model, dataset)


if __name__ == "__main__":
    wavenet_example()

