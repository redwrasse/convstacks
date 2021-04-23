from wavenetlike.dataset import AR2
import ops
import train
from models import build_ar2


def ar2_example():
    # TODO("logger.info convolution weight agreements w/ar2")
    model = build_ar2()
    dataset = AR2(cutoff=10)
    train.train_stack_ar(model,
                         dataset,
                         loss_type=ops.Losses.mse)


if __name__ == "__main__":
    ar2_example()

