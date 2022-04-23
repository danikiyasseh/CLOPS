# Continual Learning of Networks with CLOPS

CLOPS is a framework that allows neural networks to continually learn from clinical data streaming in over time. 

This repository contains the PyTorch implementation of CLOPS. For details, see **A clinical deep learning framework for continually learning from cardiac signals across diseases, time, modalities, and institutions**.
[[Nature Communications Paper](https://www.nature.com/articles/s41467-021-24483-0)], [[blogpost](https://danikiyasseh.github.io/blogs/CLOPS/)], 

## Requirements

The CLOPS code requires the following:

* Python 3.6 or higher
* PyTorch 1.0 or higher

## Datasets

### Download

The datasets can be downloaded from the following links:

1) [PhysioNet 2020](https://physionetchallenges.github.io/2020/)
2) [Chapman](https://figshare.com/collections/ChapmanECG/4560497/2)
3) [Cardiology](https://irhythm.github.io/cardiol_test_set/)

### Pre-processing

In order to pre-process the datasets appropriately for CLOPS, please refer to the following [repository](https://github.com/danikiyasseh/loading-physiological-data)


## Training

To train the model(s) in the paper, run this command:

```
python run_experiments.py
```

## Evaluation

To evaluate the model(s) in the paper, run this command:

```
python run_experiments.py
```


