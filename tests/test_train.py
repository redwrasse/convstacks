import unittest
import wavenetlike.train as train
from wavenetlike.models import build_wavenet_toy
import wavenetlike.dataset as dataset


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.model = build_wavenet_toy()
        self.data = dataset.SpeechCommands(cutoff=5).get_dataset()

    def test_train(self):
        train.train(self.model,
                    self.data,
                    nepochs=1)


if __name__ == '__main__':
    unittest.main()

