from wavenetlike.analyzers import DatasetAnalyzer

from wavenetlike.dataset import get_dataset,\
    TorchAudioDataset


def analyzer_example():
    dataset = TorchAudioDataset("SPEECHCOMMANDS")
    data_analyzer = DatasetAnalyzer(dataset)
    data_analyzer.analyze_dataset()




analyzer_example()