import dataset
import train
from models import build_wavenet


def wavenet_example():

    model = build_wavenet()
    data = dataset.SpeechCommands(cutoff=100).get_dataset()
    train.train(model, data)


if __name__ == "__main__":
    wavenet_example()

