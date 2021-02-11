import dataset
import train
from models import build_wavenet_toy

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def wavenet_toy_example():

    model = build_wavenet_toy()
    data = dataset.SpeechCommands(cutoff=5).get_dataset()
    train.train(model,
                data,
                epoch_save_freq=20)


if __name__ == "__main__":
    wavenet_toy_example()



