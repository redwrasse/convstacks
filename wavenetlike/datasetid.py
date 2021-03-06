# dataset.py
import wavenetlike.ops as ops
import torchaudio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


DATASETS = {
  
    #"CMUARCTIC": torchaudio.datasets.CMUARCTIC,
    "COMMONVOICE": torchaudio.datasets.COMMONVOICE,
    #"GTZAN": torchaudio.datasets.GTZAN,
    "SPEECHCOMMANDS": torchaudio.datasets.SPEECHCOMMANDS
}

# todo("separate logical dataset definition from dataset loader;
# dataset loader is a function of whether using gpu/cpus")
# for example setting num_workers


def get_torchaudio_dataset(key):
    logger.info('downloading dataset ....')
    project_root = Path(__file__).parent.parent
    dataset = DATASETS[key](project_root, download=True)
    return dataset


# def load_dataset(dataset, batch_size, shuffle=True):
#     return torch.utils.data.DataLoader(dataset,
#                                        batch_size=batch_size,
#                                        shuffle=shuffle)


class DataSetId:

   def get_dataset(self):
       pass


class TorchAudioDataSetId(DataSetId):

    def __init__(self, key):
        self.key = key

    def get_dataset(self):
        return get_torchaudio_dataset(self.key)


    #
    # def __init__(self, key, cutoff=None,
    #              batch_size=1, shuffle=True):
    #     super(TorchAudioDataset, self).__init__()
    #     self.key = key
    #     self.cutoff = cutoff
    #     self.batch_size = batch_size
    #     self.shuffle = shuffle
    #     self.dataset = None
    #
    # def __iter__(self):
    #     if not self.dataset: self.__get_dataset()
    #     loader = load_dataset(self.dataset,
    #                           self.batch_size,
    #                           self.shuffle)
    #     for i, e in enumerate(loader):
    #         if i == self.cutoff: return
    #         yield e
    #
    # def __get_dataset(self):
    #     self.dataset = get_dataset(self.key)


class AR2(DataSetId):

    def __init__(self,
                 params=(-0.4, 0.5),
                 init_values=(50, 60),
                 cutoff=1000):
        super(AR2, self).__init__()
        self.ar2 = ops.ar2_process(params, init_values)
        self.cutoff = cutoff
        # set a cutoff if none
        if not self.cutoff: self.cutoff = 100

    def __iter__(self):
        for i, e in enumerate(self.ar2):
            if i == self.cutoff: return
            yield e



