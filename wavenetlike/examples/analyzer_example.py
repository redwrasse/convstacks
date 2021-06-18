from wavenetlike.analyzers import DatasetAnalyzer
from wavenetlike.dataset import TorchAudioDataset


def analyzer_example():
    dataset = TorchAudioDataset("SPEECHCOMMANDS")
    data_analyzer = DatasetAnalyzer(dataset)
    data_analyzer.analyze_dataset()
    analysis_res = data_analyzer.get_analysis_result()
    print(analysis_res)


if __name__ == "__main__":
    analyzer_example()
