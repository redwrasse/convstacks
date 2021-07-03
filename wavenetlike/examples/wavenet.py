from wavenetlike.train import Trainer
from wavenetlike.models import build_wavenet
from wavenetlike.datasetid import TorchAudioDataSetId


def wavenet_example():
    model = build_wavenet()
    dataset_id = TorchAudioDataSetId(key="SPEECHCOMMANDS")
    trainer = Trainer(model, dataset_id)
    trainer.train(batch_size=1)


if __name__ == "__main__":
    wavenet_example()

