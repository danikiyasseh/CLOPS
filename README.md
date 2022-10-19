Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Continual Learning of Physiological Signals

CLOPS is a framework that allows neural networks to continually learn from clinical data streaming in over time. 

This repository contains the PyTorch implementation of CLOPS. For details, see **A clinical deep learning framework for continually learning from cardiac signals across diseases, time, modalities, and institutions**.
[[Nature Communications Paper](https://www.nature.com/articles/s41467-021-24483-0)], [[blogpost](https://danikiyasseh.github.io/blogs/CLOPS/)]

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

## Citing

If you use our code in your research, please consider citing with the following BibTex.
```text
@article{kiyasseh2021clinical,
  title={A clinical deep learning framework for continually learning from cardiac signals across diseases, time, modalities, and institutions},
  author={Kiyasseh, Dani and Zhu, Tingting and Clifton, David},
  journal={Nature Communications},
  volume={12},
  number={1},
  pages={1--11},
  year={2021},
  publisher={Nature Publishing Group}
}
```

