from wavenetlike.analyzers import TorchAudioDatasetAnalyzer

from wavenetlike.dataset import TorchAudioDataset


def analyzer_example():
    dataset = TorchAudioDataset(key="SPEECHCOMMANDS",
                                cutoff=None,
                                batch_size=1)
    data_analyzer = TorchAudioDatasetAnalyzer(dataset)
    data_analyzer.analyze_dataset()


if __name__ == '__main__':
    analyzer_example()
