from wavenetlike.train import Trainer
from wavenetlike.models import build_wavenet
from wavenetlike.dataset import TorchAudioDataset


def wavenet_example():
    model = build_wavenet()
    dataset = TorchAudioDataset(key="SPEECHCOMMANDS",
                                cutoff=100)
    trainer = Trainer(model, dataset)
    trainer.train()
    #train.train(model, dataset)


if __name__ == "__main__":
    wavenet_example()

