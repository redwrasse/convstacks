# analyzers.py

"""

Analyzers
    - ModelAnalyzer: analysis of trained model parameters and architecture
    - DatasetAnalyzer: analysis of audio data to suggest/auto-generate model
      architecture to be trained

Note:
    Follow torchaudio naming/analysis conventions described in the
    'conventions' section of https://github.com/pytorch/audio

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
        self.dataset_analysis = None
        # self.sample_rate = None
        # self.num_channels = None
        # self.min_waveform_length = None
        # self.max_waveform_length = None



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
        raise NotImplementedError


class TorchAudioDatasetAnalyzer(DatasetAnalyzer):

    def __init__(self, dataset):
        super().__init__(dataset)

    def analyze_dataset(self, cutoff=100):

        sample_rt = None
        min_waveform_len = None
        max_waveform_len = None
        num_channels = None

        logger.info("analyzing dataset ...")
        for i, e in enumerate(self.dataset):
            if i > cutoff: break
            waveform, sample_rate, *rest, = e
            batch_size, n_channels, length = waveform.shape
            sample_rate = sample_rate.data.cpu().numpy()[0]
            if not sample_rt: sample_rt = sample_rate
            if not min_waveform_len: min_waveform_len = length
            if not max_waveform_len: max_waveform_len = length
            if not num_channels: num_channels = n_channels

            min_waveform_len = min(min_waveform_len, length)
            max_waveform_len = max(max_waveform_len, length)

        logger.info("finished analyzing dataset")
        self.dataset_analysis = DatasetAnalysis(
            sample_rate=sample_rt,
            num_channels=num_channels,
            min_waveform_len=min_waveform_len,
            max_waveform_len=max_waveform_len
        )
        logger.info(f'dataset analysis: {self.dataset_analysis.report()}')


class DatasetAnalysis:

    """
    Provisional analysis quantities

       - waveform length
       - sampling frequency
       - number of channels (aside from possible discretization)
       - data type (usually 16-bit integers = 65,536 possible values?)
       - estimate of actual/needed correlation length,
       perhaps as function of distance?

    """
    def __init__(self,
                 sample_rate,
                 num_channels,
                 min_waveform_len,
                 max_waveform_len):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.min_waveform_len = min_waveform_len
        self.max_waveform_len = max_waveform_len

    def report(self):
        return self.__dict__



