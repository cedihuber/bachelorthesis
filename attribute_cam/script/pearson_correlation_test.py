import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
from CelebA.perturb_protocol.list_names import list_names
import torchvision

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from skimage.transform import resize
import torch
import torch.nn.functional as F
from PIL import Image
import shutil
import random
import pytorch_grad_cam
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats

import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torchvision
import cv2
import torch
from PIL import Image
from scipy.stats import pearsonr

#from get_shifted_landmarks import get_shifted_landmarks_df

def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Goes through the dataset and predicts the attributes")
    parser.add_argument(
        '-w',
        '--which-set',
        default="test",
        choices=("validation", "test"),
        help="Select to process the given part(s) of the dataset")
    parser.add_argument(
        '-s',
        '--source-directory',
        default='/local/scratch/datasets/CelebA/aligned_224x224',
        help="Select directory containing the input dataset")
    parser.add_argument(
        '-p',
        '--protocol-directory',
        default='CelebA/protocol',
        help=
        "Select directory containing the original filelists defining the protocol and ground truth of CelebA"
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        default="../../../../local/scratch/chuber/result/corrRise_masks_black_10batchs_30size_500masks",
        help="Path to folder where the output should be stored")
    
    parser.add_argument('-i',
                        '--image-count',
                        type=int,
                        help="if given, limit the number of images")
    parser.add_argument('-m',
                        '--model-type',
                        default='balanced',
                        choices=['balanced', 'unbalanced'],
                        help="Can be balanced or unbalanced")
    parser.add_argument(
        '--gpu',
        action="store_false",
        help='Do not use GPU acceleration (will be **disabled** when selected)'
    )
    parser.add_argument(
        '-percentage',
        '--percentage',
        default=0.5, # 0.25 not so good results
        type=float,
        help='How big is the part of a mask'
    )
    parser.add_argument(
        '-masks',
        '--masks',
        default=1000,
        type=int,
        help='Number of masks per image'
    )
    parser.add_argument(
        '-att',
        '--attributes',
        default=40,
        type=int,
        help='Number of masks per image'
    )
    
    args = parser.parse_args()

    return args


def pearson_correlation_multi(x, y): # x shape (500,40) y shape (500,50176)
    # pearson correlation = sum ( (x - x.mean) * (y-y. mean) ) / sq_root( sum( (x-x.mean)^2 ) * sum( (y-y.mean)^2 ) )
    x = x - x.mean(dim=0, keepdim=True)  # (N, A)
    y = y - y.mean(dim=0, keepdim=True)  # (N, M)
    nominater = torch.matmul(x.T, y)
    #print(f'x = {x.shape}, y = {y.shape}')
    x_norm = torch.norm(x, dim=0, keepdim=True)  # (1, 40)
    y_norm = torch.norm(y, dim=0, keepdim=True)  # (1, 50176)
    #print(f'x_norm = {x_norm.shape}, y_norm = {y_norm.shape}')
    denom = torch.matmul(x_norm.T, y_norm)  # (A, M)
    denom[denom == 0] = 1e-8

    corr = nominater / denom  # (A, M)
    return corr




def main():
    torch.manual_seed(0)
    x = torch.randn(500, 40)
    y = torch.randn(500, 50176)

    # Compute correlation with your function
    corr_torch = pearson_correlation_multi(x, y)  # shape (40, 50176)

    # Choose random indices to validate
    a_idx = 3        # e.g., 3rd feature in x
    m_idx = 1000     # e.g., 1000th feature in y

    # Get corresponding x[:, a_idx] and y[:, m_idx] for comparison
    x_np = x[:, a_idx].numpy()
    y_np = y[:, m_idx].numpy()

    # Scipy correlation
    corr_scipy, _ = pearsonr(x_np, y_np)

    # Your function's correlation
    corr_custom = corr_torch[a_idx, m_idx].item()

    print(f"Pearson Correlation (scipy):  {corr_scipy:.6f}")
    print(f"Pearson Correlation (custom): {corr_custom:.6f}")
    print(f"Difference: {abs(corr_scipy - corr_custom):.6e}")

if __name__ == '__main__':
    main()
