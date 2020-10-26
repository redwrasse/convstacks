

Background
~~~~~~~~~~

Convolutions are translation-invariant kernels: a dog in a picture is picked up whether it is 27 or 142 pixels from the left edge of the scene. The convolution :math:`\mathbf{x} * \mathbf{w}` between a vector :math:`\mathbf{x}` and a kernel :math:`\mathbf{w}` is defined :math:`s_i = x_l w_{i-l}`. A convolutional layer of a neural network is a convolution with a bias and activation, :math:`\phi(\mathbf{x} * \mathbf{w} + \mathbf{b})`.

Properties
~~~~~~~~~~

Composition
=========

Convolutions can be composed. In fact they usually are in neural networks. For example, stacks of increasingly dilated convolutions underpin the Wavenet model, yielding efficient long-range correlations.

Kernel Size
=========

Dilation
=========

Stride
=========

Tiling
=========

Padding
=========



