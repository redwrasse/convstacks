# analyzer.py


def analyze_stack(stack):
    # ought to determine effective kernel length
    # for loss function
    # the effective kernel length is defined as the longest previous dependence
    # aka if x_n = f(x_(n-k), x_(n-k+1),...) then the effective kernel length is k
    # use partial derivatives
    effective_kernel_length = sum(stack.kernel_length * stack.dilation_rate**i
                                  for i in range(stack.n_layers))
    print(f'stack with effective kernel length of {effective_kernel_length}')


def analyze_dataset(dataset):
    pass

