# CLOPS

CLOPS is a replay-based continual learning strategy that employs both a buffer storage and acquisition mechanism. 

This method is described in: "CLOPS: Continual Learning for Physiological Signals"

# Requirements

The CLOCS code requires

* Python 3.6 or higher
* PyTorch 1.0 or higher

# Datasets

## Download

The datasets can be downloaded from the following links:

1) PhysioNet 2020: https://physionetchallenges.github.io/2020/
2) Chapman: https://figshare.com/collections/ChapmanECG/4560497/2
3) Cardiology: https://irhythm.github.io/cardiol_test_set/

## Pre-processing

In order to pre-process the datasets appropriately for CLOPS, please refer to the following repository:
https://anonymous.4open.science/r/9ecc66f3-e173-4771-90ce-ff35ee29a1c0/

# Training

To train the model(s) in the paper, run this command:

```
python run_experiments.py
```

# Evaluation

To evaluate the model(s) in the paper, run this command:

```
python run_experiments.py
```


