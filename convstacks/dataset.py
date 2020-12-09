# dataset.py
import ops


class Dataset:
    def __init__(self, name):
        self.name = name


class SpeechCommands(Dataset):

    def __init__(self, name, cutoff):
        super(SpeechCommands, self).__init__(name)
        self.dataset = ops.download_sample_audio(cutoff=cutoff)


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
