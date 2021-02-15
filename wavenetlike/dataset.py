# dataset.py
import wavenetlike.ops as ops


class Dataset:
    def __init__(self, name):
        self.name = name

    def get_dataset(self):
        pass


class SpeechCommands(Dataset):

    def __init__(self, cutoff):
        super(SpeechCommands, self).__init__("Speech Commands")
        self.dataset = ops.download_sample_audio(cutoff=cutoff)

    def get_dataset(self):
        return self.dataset


class AR2(Dataset):

    def __init__(self):
        super(AR2, self).__init__("AR2")
        a, b = -0.4, 0.5
        x0, x1 = 50, 60
        n_samples = 100
        data = []
        gen = ops.ar2_process(a, b, x0, x1)
        for i in range(n_samples):
            data.append(gen.__next__())
        self.dataset = data

    def get_dataset(self):
        return self.dataset

