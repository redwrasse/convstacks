import wavenetlike.train as train
from wavenetlike.models import build_wavenet_toy
from wavenetlike.dataset import Dataset



def wavenet_toy_example():

    model = build_wavenet_toy()
    dataset = Dataset(key="SPEECHCOMMANDS",
                      cutoff=5)
    train.train(model,
                dataset,
                epoch_save_freq=5)


if __name__ == "__main__":
    wavenet_toy_example()



