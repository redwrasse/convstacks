from wavenetlike.train import Trainer
from wavenetlike.models import build_wavenet_toy
from wavenetlike.datasetid import TorchAudioDataSetId


def wavenet_toy_example():

    model = build_wavenet_toy()
    dataset = TorchAudioDataSetId(key="SPEECHCOMMANDS")
    trainer = Trainer(model, dataset)
    trainer.train(batch_size=1)


if __name__ == "__main__":
    wavenet_toy_example()



