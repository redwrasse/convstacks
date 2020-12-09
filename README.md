# wavenetlike

**Note: This library is a work in progress. To train the original Wavenet on the Speech Commands dataset, run this [script](https://github.com/redwrasse/convstacks/blob/main/convstacks/wavenet_example.py), or [this one](https://github.com/redwrasse/convstacks/blob/main/convstacks/wavenet_example_tpu.py) for TPUs. To access other library functionality currently work directly with source (examples below).** 

### Background
A library for building wavenet-like models: generative auto-regressive models with large receptive fields. The canonical example is of course [Wavenet](https://arxiv.org/pdf/1609.03499.pdf) itself.

Wavenet is just one in a family of models providing long receptive fields with a reasonable number of parameters. The specific model depends on the data and use case.

This library is built on the [Pytorch API](https://pytorch.org/docs/stable/index.html).

### Use
Intended workflow
```
import wavenetlike

dataset = Dataset(source)
analyze(dataset)
    waveform lengths
    sample frequency
    correlation lengths
    ...
# perhaps suggest model params based
# off dataset analysis
model = build_wavenetlike(params)
analyze(model)
    total num parameters / depth
    receptive field size
    ...

train(model, dataset)
predict(model, inputs)
```

### Docs

Docs are built with [Sphinx](https://www.sphinx-doc.org/en/master/). For the time being please build docs locally, following this [script](./update_docs).
