from wavenetlike.analyzers import DatasetAnalyzer

from wavenetlike.dataset import TorchAudioDataset


def analyzer_example():
    dataset = TorchAudioDataset("SPEECHCOMMANDS")
    data_analyzer = DatasetAnalyzer(dataset)
    data_analyzer.analyze_dataset()


if __name__ == "__main__":
    analyzer_example()
