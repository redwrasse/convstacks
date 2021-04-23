import unittest
import torchaudio

import wavenetlike.dataset as dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        pass

    def test_torchaudio_datasets(self):
        """ check all torchaudio datasets actually exist """
        for attr in dataset.DATASETS:
            assert hasattr(torchaudio.datasets, attr)
