import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
from CelebA.perturb_protocol.list_names import list_names
import torchvision
import numpy as np

def main():
    scores_directly = {}  # Store image names and their corresponding scores
    with open("result/myresult/RISE/Prediction-perturb2.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]  # First column contains the perturbed image name
            scores = np.array(row[1:], dtype=np.float32)  # Convert scores to float
            scores_directly[image_name] = scores  # Store in dictionary
        # Generate saliency map
    scores_saved = {}  # Store image names and their corresponding scores
    with open("result/myresult/RISE/Prediction-perturb.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]  # First column contains the perturbed image name
            scores = np.array(row[1:], dtype=np.float32)  # Convert scores to float
            scores_saved[image_name] = scores  # Store in dictionary
        # Generate saliency map
    #print(scores_saved)
    difference = 0
    count = 0
    for image, values_directly in scores_directly.items():
        values_saved = scores_saved[image]
        for i in range(0, len(values_directly)-1):
            difference += (values_directly[i] - values_saved[i]) **2
            count += 1
    return difference / count

if(__name__ == '__main__'):
    print(main())