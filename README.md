# Dual-Region Consistency Learning with Contrastive Refinement

This repository contains the code implementation for the paper "Dual-Region Consistency Learning with Contrastive Refinement for semi-supervised medical image segmentation".

## Requirements

List of dependencies required to run the code.

- python
- numpy
- torch
- h5py
- nibabel
- scipy
- skimage
- tqdm
- medpy

These dependencies can be installed using the following command:

    pip install -r requirements.txt

## Dataset

The dataset used for this project can be downloaded from the following links:

- https://www.cardiacatlas.org/atriaseg2018-challenge/
- https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

## How to train

1. Clone the repository
2. Install the required packages using the command mentioned above
3. Download the dataset from the link provided above and extract it to the data/ directory
To train a model,
```
python Pancreas_train.py  #for Pancreas training
python LA_train.py  #for LA training
```
To test a model,
```
python Pancreas_test.py  #for Pancreas testing
python LA_test.py  #for LA testing
```
### Acknowledgment

The development of this project is based on [SCC](https://github.com/PerceptionComputingLab/SCC)

