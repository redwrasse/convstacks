

Background
============
A library for building causal convolutional networks with large receptive fields by stacked dilated convolutions (like Wavenet).

The library will analyze properties of the given dataset, and help select an appropriate network.
The network should have a sufficiently long-range receptive field size and ideally not too many parameters to train.

The library includes some motivating datasets.


Concepts
============
A convolution is

A dilated convolution is

Notes about properties of audio: sample rate, causal length, timeseries length, # of channels.

Effective receptive field of stacked dilated convolutions...

Usage
============

First, select a dataset. The analyzer will then provide properties of that dataset.
Next, the library will help select an appropriate network. Finally the network can be trained, and
the results tested and observed.

Add some notes about training: loss, accuracy, training time.

Datasets
============
All datasets to be trained are encapsualated in a Dataset object. As stated above some example Datasets are provided:

* An AR(2) autoregressive model, :math:`X_t = c + a X_{t-1} + b X_{t-2} + \epsilon_t`. The parameters :math:`a, b` will be the network weights in a single trained convolutional layer of kernel length 2.
* B
* C

Examples
============

Usage of example datasets.

AR(2) model

.. code-block:: python

   Some python code.