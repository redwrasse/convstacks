# analyzers.py

"""

Analyzers
    - ModelAnalyzer: analysis of trained model parameters and architecture
    - DatasetAnalyzer: analysis of audio data to suggest/auto-generate model
      architecture to be trained


"""
import logging

logger = logging.getLogger(__name__)


class ModelAnalyzer:

    def __init__(self, model):
        self.model = model

    def analyze_model(self):
        """
           to return
               - receptive field length
               - ratio of depth / total # of parameters

           :param model:
           :return:
           """
        pass


class DatasetAnalyzer:

    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_dataset(self):
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



