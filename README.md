# CAM visualizations of attribute classifiers

This code runs and evaluates different CAM techniques on two different attribute classifiers.
For creating the CAMs, we directly rely on functionality from [PyTorch-Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).

## Installation

This code comes with a conda installation, which you can use via:

    conda env create -f environment.yaml
    conda activate attribute-cam

This will install several runnable scripts into the conda environment `attribute-cam`.

## Scripts

Implementations of the following scripts can be found in `attribute_cam/script`. All scripts contain a `--help` option to show possible arguments.

* `predict.py` computes the prediction of the model on the original frontal images. This is rather quick.

* `extract_cams.py` extracts CAM images for the frontal images of the given model and CAM technique and all attributes. This might take a while.

* `average_cams.py` will compute average CAM images using various filter criteria, based on the ground truth and the prediction for each attribute.

* `analyze.py` evaluates mean (and std) of the acceptable mask ratio measure over all attributes for various filter criteria.


## Code

The following code files are available in `attribute_cam`:

* `dataset.py` implements the dataset and the image IO

* `model.py` implements the original AFFACT models and the atribute prediction

* `cam.py` implements the extraction of CAMs from the images for all attributes, and the averaging of CAM images

* `filter.py` implements IO of prediction and ground-truth attributes, and defines several filters that make use of the prediction and the ground truth

* `evaluation.py` computes acceptable mask ratios and error rates

* `mask.py` defines precise masks for all attributes that are required to compute the acceptable mask ratio

## Additional files

* `files` contains several files defining the frontal subset, the ground truth, and landmarks

* `helper_scripts` left-overs from the original thesis that might be removed later

* `additional_scripts` scripts to compute frontal subsets, eralier versions of the masks, and alike
