# Bachelor Thesis Johanna Bieri (09.08.2023)

The code was run on a computer with a windows 11 OS and a NVIDIA GeForce RTX 3060 graphics card. The images of the CelebA dataset used have the size 224x224 (not contained here).

The code was updated to work on Linux systems (which it did not initially), and some speed-ups were implemented.

## Conda Environment:

The file "bieri_env.yml" contains the conda environment with all the necessary packages for running the code. It can be installed by running "conda env create -f environment.yml". There might be some conflicts with the required packages when trying to install it on Linux or macOS.


## Code:

folder code_core:
1. Run the analyze.py file with the required arguments. This will generate the GradCAMs for every input image and compute the Acceptable Mask Ratio and other possibly useful data and store it in a csv file in the same folder. The error rate is computed as well and stored in a separate csv file.

folder helper_scripts:
2. By running the cam_avg.py file several different average GradCAMs are computed and stored at the given location.

3. By running the files starting with "get_amr_avg" different kinds of averages of the amr can be computed and stored at the given location.

4. By running the kl_distance.py file the Kullback-Leibler distance for the average GradCAMs from the balanced to the average GradCAMs of the unbalanced network can be computed and stored at the given location.

5. By running the covmat.py file the covariance matrix for two given arrays of values (error rate and average acceptable mask ratio) can be computed.

folder additional_scripts:
This folder contains additional scripts. The masks version 1 (masks.py), the scripts for shifting the landmarks and getting them(shift_landmarks.py, get_landmarks.py, get_shifted_landmarks.py) and the scripts for filtering out the non-frontal images from the CelebA dataset (filter_imgs.py, list_imgs.py).

files:
All the files needed are contained in this folder.
