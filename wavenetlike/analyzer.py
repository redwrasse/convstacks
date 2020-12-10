# analyzer.py

"""
 Analysis of the data to suggest possible/autogenerate model

 Analysis of the model

"""


def analyze_model(model):
    """
    to return
        - receptive field length
        - ratio of depth / total # of parameters

    :param model:
    :return:
    """
    pass


def analyze_dataset(dataset):
    """
    to return
        - waveform length
        - sampling frequency
        - number of channels (aside from possible discretization)
        - data type (usually 16-bit integers = 65,536 possible values?)
        - estimate of actual/needed correlation length,
        perhaps as function of distance?

    :param dataset:
    :return:
    """
    pass

