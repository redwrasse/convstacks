import dataset
import ops
import train
from models import build_ar2


def ar2_example():
    # TODO("logger.info convolution weight agreements w/ar2")
    model = build_ar2()
    data = dataset.AR2().get_dataset()
    train.train_stack_ar(model, data, loss_type=ops.Losses.mse)


if __name__ == "__main__":
    ar2_example()

