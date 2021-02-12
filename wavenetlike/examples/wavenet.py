import wavenetlike.dataset as dataset
import wavenetlike.train as train
from wavenetlike.models import build_wavenet


def wavenet_example():

    model = build_wavenet()
    data = dataset.SpeechCommands(cutoff=100).get_dataset()
    train.train(model, data)


if __name__ == "__main__":
    wavenet_example()

