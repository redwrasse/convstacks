# examples.py
"""
Assuming a finite past time dependence and linear dependence
it becomes an auto-regressive model. This can be trained by
convolutional layers but the time series has to be appropriately
lined up at input and output layers.
The general rules (assuming a left-padding).
# 1) shift output left one index (so an index isn't tied to itself)
        # 2) ignore first k - 1 indices
        # 3) ignore last index in both input and output
        # aka output is the range k-1:-1,
        # input is the range k:
Note this means the kernel size is actually one less than what one may expect,
since an index isn't tied to itself.
What it looks like in this example with k = 2. The first k - 1 (= 1 for k = 2 in this case)
 and last index are ignored for calculating the loss.
    (ig)            (ig)
    x1  x2  x3  x4  x5
    *   *   *   *   *
/   | / | / | / | / |
    *   *   *   *   *
    x0  x1  x2  x3  x4
   (ig)            (ig)
Example auto-regressive model:
    AR(2) process x_t = a x_t-1 + b xt-2 + noise
    stationary if a in [-2, 2], b in [-1, 1]
This trained model then allows prediction, outputting the next timestep value x5. Iterate to
generate a sequence of predictions.
"""
from utils import ar2_process
from stack import Stack, train_stack_ar, analyze_stack


def train_ar2():
    stack = Stack(n_layers=1, kernel_length=2, dilation_rate=1)
    analyze_stack(stack)
    a, b = -0.4, 0.5
    x0, x1 = 50, 60
    n_samples = 100
    data = []
    gen = ar2_process(a, b, x0, x1)
    for i in range(n_samples):
        data.append(gen.__next__())

    train_stack_ar(stack, data)


if __name__ == '__main__':
    train_ar2()

