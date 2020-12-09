

Background
============

A library for building wavenet-like models: generative auto-regressive models with large receptive fields.
The canonical example is of course `Wavenet <https://arxiv.org/pdf/1609.03499.pdf>`_ itself.

Wavenet is just one in a family of models providing
long receptive fields with a reasonable number of parameters. The specific model depends on the data and use case.

This library is built on the `Pytorch API <https://pytorch.org/docs/stable/index.html>`_.

Use
============

.. code-block:: python

    import wavenetlike

    dataset = Dataset(source)
    analyze(dataset)
        waveform lengths
        sample frequency
        correlation lengths
        ...
    # perhaps suggest model params based
    # off dataset analysis
    model = build_wavenetlike(params)
    analyze(model)
        total num parameters / depth
        receptive field size
        ...

    train(model, dataset)
    predict(model, inputs)


Concepts
============
There are a large number of possible architecture choices and parameter tweaks, but a few
are of primary interest

* waveform sample frequency and expected correlation length.
* :math:`\gamma :=` ratio of the total # of parameters / network depth.
* :math:`\Delta :=` receptive field length of the network.

At its core Wavenet-like models are just about combining increasingly dilated convolutions to generate
autoregressive models with large :math:`\Delta` and small  :math:`\gamma`. The rest are possible architectural tricks like
gated activations, skip-residual connections, and waveform discretization.

The original Wavenet as published trains on audio with sample frequencies of ~16,000 samples/second,
with multiple (3-5) blocks of dilated convolutions generating a receptive field on the order of ~200-300ms,
or a receptive field length of ~3200-4800 samples.


For possible architecture choices and parameter tweaks see separate page (tbd).

For a discussion of technical and mathematical details see separate page (tbd).

Analysis
============
This library intends to provide two forms of analysis to aid with practically building and training a suitable model

i) analysis of the dataset: some sense of correlation length
ii) analysis of the model: receptive field length, total # of parameter/network depth.


Ideally it will suggest a model from analysis of the dataset, or at least point the user in the right direction.

Training
============
Training the original Wavenet on real audio takes a long time (n hours?, deps on computational resource).
On the other hand training a conceptually equivalent smaller model can be done quickly, for
demonstration purposes. See WavenetConstants and WavenetToyConstants.

Because of the translational invariance for an autoregressive model, training is done in parallel for each :math:`p(x_i \mid x_{i -\Delta} ... x_{i-1})`, with :math:`\Delta` the receptive field size.

Datasets
============
Formal datasets provided

* An AR(2) autoregressive model, :math:`X_t = c + a X_{t-1} + b X_{t-2} + \epsilon_t`. Parameters :math:`a, b` will match network weights in a single trained convolutional layer of kernel length 2.
* The `Speech Commands <https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html>`_ audio dataset.
* Custom Dataset: load custom waveforms.

Examples
============

* The AR(2) model can be learned immediately with a single convolutional layer.
.. code-block:: python

    # current form
    stack = Block(n_layers=1, kernel_length=2, dilation_rate=1)
    a, b = -0.4, 0.5
    x0, x1 = 50, 60
    n_samples = 100
    data = []
    gen = ar2_process(a, b, x0, x1)
    for i in range(n_samples):
        data.append(gen.__next__())
    train_stack_ar(stack, data, loss_type=Losses.mse)


* A smaller wavenet model (fewer parameters and coarser discretization) achieves ~50% training accuracy in a minute on the `Speech Commands <https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html>`_  dataset. Of course, its predictions with be commensurably coarse.

.. code-block:: python

    # current form
    model = build_wavenet_toy()
    dataset = ops.download_sample_audio(cutoff=5)
    train.train(model, dataset)


* Full Wavenet.
.. code-block:: python

    # current form
    model = build_wavenet()
    dataset = ops.download_sample_audio(cutoff=5)
    train.train(model, dataset)


To Do
============

This library is a work in progress. Some future tasks

* Support conditioning on a variable (see  `Wavenet paper <https://arxiv.org/pdf/1609.03499.pdf>`_)
* Support easy distributed training on TPUs.
* Provide an intelligent workflow from dataset to model selection.