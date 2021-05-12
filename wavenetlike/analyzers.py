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
        self.sample_rate = None
        self.min_waveform_length = None
        self.max_waveform_length = None

    def __repr__(self):
        repr = f'''
dataset analyzer
----------------        
sample rate: {self.sample_rate}
waveform length: {self.min_waveform_length} to {self.max_waveform_length}
        '''
        return repr

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
        logger.info("analyzing dataset ...")
        # todo("analyzer, wip. hack first analyzer below")
        i = 0
        n = 100
        mx_waveform_len = -float('inf')
        mn_waveform_len = float('inf')
        for e in self.dataset:
            if i >= n: break
            waveform, sample_rate, *args = e
            self.sample_rate = int(sample_rate[0])
            wave_len = waveform.shape[-1]
            mx_waveform_len = max(wave_len, mx_waveform_len)
            mn_waveform_len = min(wave_len, mn_waveform_len)
            i += 1

        self.max_waveform_length = mx_waveform_len
        self.min_waveform_length = mn_waveform_len
        logger.info(f"finished analyzing dataset:\n{self}")





