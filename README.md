# Visualization of Facial Attribute Classifiers via Perturbation-based Methods

This is a fork of (https://github.com/AIML-IfI/attribute-cam) and extends its functionality by adding an adapted CorrRISE method. As such this readme describes both the original functionality as well as the additionally implemented features.

# Biased Binary Attribute Classifiers Ignore the Majority Classes

This package provides all implementations of methods and scripts to reproduce the results of our paper `Biased Binary Attribute Classifiers Ignore the Majority Classes` to be presented at the [Swiss Conference on Data Science (SDS) 2024](https://sds2024.ch):

If you are making use of the source code for your own experiments, please cite the following:

    @inproceedings{zhang2024attribute,
        author = {Zhang, Xinyi and Bieri, Johanna Sophie and G\"unther, Manuel},
        title = {Biased Binary Attribute Classifiers Ignore the Majority Classes},
        booktitle = {Swiss Conference on Data Science (SDS)},
        year = {2024}
    }

This code runs and evaluates different CAM techniques on two different attribute classifiers.
For creating the CAMs, we directly rely on functionality from [PyTorch-Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).
It is based on the [Bachelor thesis of Johanna Bieri](https://www.merlin.uzh.ch/publication/show/24056) in the [Artificial Intelligence and Machine Learning](https://www.ifi.uzh.ch/en/aiml.html) group at the [Department of Informatics, University of Zurich](https://www.ifi.uzh.ch/en.html).
Additionally there is code of an adapted CorrRISE method for facial attribute classification to explain the same classifiers.

## Installation

This code comes with a conda installation, which you can use via:

    conda env create -f environment.yaml
    conda activate attribute-cam

This will install several executable scripts into the conda environment `attribute-cam`.
Additionally, this will automatically install this package in editable mode via `pip`.

## Pre-trained Models

In this package, we evaluate two facial attribute prediction models, which have been trained within the Caffe framework, and which were converted to PyTorch and slightly adapted to work with CAM techniques.
Please download the [Pre-trained Models](https://seafile.ifi.uzh.ch/d/58644ee482d34425b5a1/) and place them into the `model` directory.

## Data

All of our evaluation relies on faces aligned to 224x224 pixels.
We strictly follow the implementation in [The Original AFFACT Paper](https://ieeexplore.ieee.org/abstract/document/8272686).
For simplicity, you can download the [Pre-aligned Faces of the CelebA dataset](https://seafile.ifi.uzh.ch/f/5853d95aef724bafa868) and extract them into the `CelebA` directory.
Required protocol files can be found within the `CelebA/protocol` directory, which are partially taken from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and partially created by ourselves.


## Scripts

Implementations of the following scripts can be found in `attribute_cam/script`. All scripts contain a `--help` option to show possible arguments.

* `predict.py` computes the prediction of the model on the original frontal images. This is rather quick.

* `extract_cams.py` extracts CAM images for the frontal images of the given model and CAM technique and all attributes. This might take a while.

* `corrRISE_masks_black.py`, `corrRISE_masks_black_white.py`, and `corrRISE_masks_blurry.py` extracts saliency maps for the frontal images of the given model and mask style and all attributes. This might take a while.

* `rise_for_facial_attribute_classifier.py` is an adaption from the original RISE method for facial attribute classifiers. Extracts saliency maps for the frontal images of the given model and mask style and all attributes. This might take a while.

* `average_cams.py` will compute average CAM images using various filter criteria, based on the ground truth and the prediction for each attribute. This requires both `predict.py` and `extract_cams.py` to have run.

* `average_perturbation.py` will compute average saliency maps using various filter criteria, based on the ground truth and the prediction for each attribute. This requires both `predict.py` and `extract_cams.py` to have run.

* `analyze.py` evaluates mean (and std) of the proportional energy measured over all attributes for various filter criteria. This requires both `predict.py` and `extract_cams.py` to have run.

* `analyze_balanced.py` evaluates mean (and std) of the proportional energy measured over all attributes for various filter criteria for balanced model. This requires both `predict.py` and one corrRISE method to have run.

* `analyze_unbalanced.py` evaluates mean (and std) of the proportional energy measured over all attributes for various filter criteria for unbalanced model. This requires both `predict.py` and one corrRISE method to have run.

* `cosine_evaluation.py` calculates the cosine similarity of averaged saliency maps of two different methods. This requires both `predict.py` and two corrRISE method or CAM methods to have run.


## Code

The following code files are available in `attribute_cam`:

* `dataset.py` implements the dataset and the image IO

* `dataset_perturb.py` implements the dataset and the image IO for the adapted CorrRISE

* `model.py` implements the original AFFACT models and the attribute prediction

* `cam.py` implements the extraction of CAMs from the images for all attributes, and the averaging of CAM images. The same can be used for averaging the saliency maps of adapted CorrRISE.

* `filter.py` implements IO of prediction and ground-truth attributes, and defines several filters that make use of the prediction and the ground truth

* `evaluation.py` computes proportional energy and error rates

* `mask.py` defines precise masks for all attributes that are required to compute the proportional energy

* `utils.py` methods used in for different CorrRISE styles to enhance modularization


## Additional Scripts

* `additional_scripts` scripts to compute frontal subsets, earlier versions of the masks, and alike


## Unused Scripts

* `unused_scripts` scripts that have been developed during the working process but were not used for final evaluation because they were not entirely correct or not completly finished.

## Troubleshooting

In case you find mistakes in our code, please either file an issue, or contact [Manuel Günther](mailto:siebenkopf@googlemail.com).
