import wavenetlike.train as train
from wavenetlike.models import build_wavenet
from wavenetlike.dataset import TorchAudioDataset


def wavenet_example():
    model = build_wavenet()
    dataset = TorchAudioDataset(key="SPEECHCOMMANDS",
                                cutoff=100)
    train.train(model, dataset)


if __name__ == "__main__":
    wavenet_example()

