from wavenetlike.dataset import AR2
import train
from models import build_ar2


def ar2_example():
    # TODO("logger.info convolution weight agreements w/ar2")
    model = build_ar2()
    dataset = AR2()
    train.train_stack_ar(model,
                         dataset)


if __name__ == "__main__":
    ar2_example()

