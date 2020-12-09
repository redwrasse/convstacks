# convstacks

A library for building stacked dilated convolutions for generative autoregressive models (like Wavenet).

wip

Usage: see docs

### Datasets

### Examples



Sample use

```python


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
	total # parameters / depth
	receptive field size
	...
	
train(model) 
predict(model, inputs)


```