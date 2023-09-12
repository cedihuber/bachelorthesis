import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
import argparse
import os
import shutil
from get_shifted_landmarks import get_shifted_landmarks_df, get_shifted_landmarks_img
from tqdm import tqdm

def calc_d(df, img_number):
    landmarks_aligned = get_shifted_landmarks_img(df, img_number)
    # center of eyes (lefteye_x+(righteye_x-lefteye_x)/2)
    x1 = landmarks_aligned[0] + (landmarks_aligned[2] - landmarks_aligned[0])/2
    y1 = (landmarks_aligned[1] + landmarks_aligned[3])/2
    p1 = (x1, y1)
    # center of mouth (leftmouth_x+(rightmouth_x - leftmouth_x)/2)
    x2 = landmarks_aligned[6] + (landmarks_aligned[8] - landmarks_aligned[6])/2
    y2 = (landmarks_aligned[7] + landmarks_aligned[9])/2
    p2 = (x2, y2)
    # tip of the nose
    p3 = (landmarks_aligned[4], landmarks_aligned[5])
    # D (distance from eye center to mout center)
    d1 = np.linalg.norm(np.cross(np.asarray(p2)-np.asarray(p1), np.asarray(p1)-np.asarray(p3)))/np.linalg.norm(np.asarray(p2)-np.asarray(p1))
    # d (distance D to nose)
    d2 = y2-y1
    
# =============================================================================
#     # draw landmarks and D on image
#     data = image.imread(r"C:\Users\Johanna\celeba\aligned_224x224_validation\162922.png")
#     plt.title(img_number)
#     # left eye
#     plt.plot(landmarks_aligned[0], landmarks_aligned[1], marker='o', color="cyan", markersize=5)
#     # right eye
#     plt.plot(landmarks_aligned[2], landmarks_aligned[3], marker='o', color="cyan", markersize=5)
#     # tip of the nose
#     plt.plot(landmarks_aligned[4], landmarks_aligned[5], marker='o', color="cyan", markersize=5)
#     # left corner mouth
#     plt.plot(landmarks_aligned[6], landmarks_aligned[7], marker='o', color="cyan", markersize=5)
#     # right corner mouth
#     plt.plot(landmarks_aligned[8], landmarks_aligned[9], marker='o', color="cyan", markersize=5)
#     # center of eyes
#     plt.plot(x1, y1, marker='o', color="yellow", markersize=3)
#     # center of mouth
#     plt.plot(x2, y2, marker='o', color="yellow", markersize=3)
#     if p1[0] > p2[0]:
#         cx1, cy1 = [p1[0], p2[0]], [p1[1], p2[1]] 
#     else:
#         cx1, cy1 = [p2[0], p1[0]], [p1[1], p2[1]] 
#     plt.plot(cx1, cy1, marker='o', color="yellow", markersize=3)
#     
#     # Hide X and Y axes tick marks
#     ax = plt.gca()
#     ax.set_xticks([])
#     ax.set_yticks([])
#     
#     plt.imshow(data)
#     
#     plt.show()
# =============================================================================
 
    return d1, d2

def main(args):
    df = get_shifted_landmarks_df(args.path_lm)  
        
    # make list with image paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    imgs = os.listdir()
    img_numbers = [int(im.split('.')[0]) for im in imgs]
    
    
    # check if output folder exists, if not, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # select frontal images (d/D <= threshold) and copy to other folder
    for nr in tqdm(img_numbers):
        d, D = calc_d(df, nr)
        if (d/D) <= args.threshold:
            source = os.path.join(input_dir, f'{nr}.png')
            dest = os.path.join(output_dir, f'{nr}.png')
            shutil.copy(source, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy frontal pictures (value below threshold) to new folder.")
    parser.add_argument(
        'input_dir',
        type=str,
        help="Path folder containing the images of the CelebA dataset (aligned_224x224)"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Path folder where the output should be stored"
    )
    parser.add_argument(
        'path_lm',
        type=int,
        help="Path to file containing the shifted landmarks (landmarks_aligned_shifted.txt)"
    )
    parser.add_argument(
        'threshold',
        type=float,
        default = 0.1,
        help="The threshold for filtering out an image, default is 0.1"
    )
    args = parser.parse_args()
    main(args)
