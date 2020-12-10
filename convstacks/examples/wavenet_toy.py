import dataset
import train
from models import build_wavenet_toy


def wavenet_toy_example():

    model = build_wavenet_toy()
    data = dataset.SpeechCommands(cutoff=5).get_dataset()
    train.train(model, data)


if __name__ == "__main__":
    wavenet_toy_example()



