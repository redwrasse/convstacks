import unittest
import wavenetlike.train as train
from wavenetlike.models import build_wavenet_toy
from wavenetlike.datasetid import TorchAudioDataSetId


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.model = build_wavenet_toy()
        self.dataset = TorchAudioDataSetId(key="SPEECHCOMMANDS",
                                           cutoff=5)

    @unittest.skip("too expensive")
    def test_train(self):
        train.train(self.model,
                    self.dataset,
                    nepochs=1)


if __name__ == '__main__':
    unittest.main()

