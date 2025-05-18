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

#from get_shifted_landmarks import get_shifted_landmarks_df
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # mit : cuda: 0 kann ich angeben auf welcher gpu nummer, gpustat um gpu usage zu schauen
print(f"Using device: {device}")  # Optional: To confirm whether GPU is used        

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
        default="../../../../local/scratch/chuber/result/minPlus_testing",
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
        default=2000,
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


def load_img(path, input_size=(224, 224)):
    image = torchvision.io.image.read_image(path)
    # convert to the required data type
    image = image / 255.0
    # add the required batch dimension
    image = image.unsqueeze(0)
    
    return image, image[0].numpy().transpose(1,2,0) 


def save_masks_as_images(masks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all masks
    for i, mask in enumerate(masks):
        # If the mask has one channel, squeeze it to remove the channel dimension
        if mask.shape[0] == 1:
            mask_np = mask.squeeze().cpu().numpy()  # Shape: (H, W)
        else:
            mask_np = mask.cpu().numpy()  # Shape: (C, H, W)

        # Normalize to 0-255 for saving as an image (works for both 1-channel and 3-channel masks)
        mask_np = (mask_np * 255).astype(np.uint8)

        # If mask is 3-channel (RGB), you need to reorder the dimensions for saving as an image
        if mask_np.shape[0] == 3:
            # Convert from (C, H, W) to (H, W, C)
            mask_np = np.transpose(mask_np, (1, 2, 0))

        # Convert to a PIL image and save
        img = Image.fromarray(mask_np)
        img.save(os.path.join(output_dir, f"mask_{i}.png"))
    
def gaussian_mask(shape, center, sigma):
    """Create a 2D Gaussian mask with given center and standard deviation (sigma)."""
    x = torch.arange(0, shape[1], dtype=torch.float32)
    y = torch.arange(0, shape[0], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    cx, cy = center
    mask = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return mask

def single_removal(image, width, steps, affact, image_name):
    print(image.shape)
    C, H, W = image.shape
    saliency_map = torch.zeros((40,H, W), dtype=torch.float32, device=device)
    original_score = affact.predict_corrrise((image.unsqueeze(0), image_name)) # 1,40
    
    for i in tqdm(range(0, H, steps)):
        for j in range(0, W, steps):
            mask = gaussian_mask((H, W), (j, i), width).to(device)  # Note (x, y) order for center
            mask = mask.unsqueeze(0).repeat(C, 1, 1)  # Match shape (C, H, W)
            
            # plt.imshow(mask.permute(1,2,0).cpu().numpy(), cmap="jet", alpha=1)
            # plt.axis("off")
            # plt.savefig(f"../../../../local/scratch/chuber/result/minPlus_testing/mask{i},{j}.png", bbox_inches='tight')
            # plt.close()
            # #print(f"G shape{mask.shape}")
            # # Apply mask (removal: multiply by 1 - G)
            perturbed_image = image * (1 - mask)
            
            #print(perturbed_image.shape)
            # plt.imshow(perturbed_image.permute(1,2,0).cpu().numpy(), cmap="jet", alpha=1)
            # plt.axis("off")
            # plt.savefig(f"../../../../local/scratch/chuber/result/minPlus_testing/perturbed_image{i},{j}.png", bbox_inches='tight')
            # plt.close()
            
            # plt.imshow(image.permute(1,2,0).cpu().numpy(), cmap="jet", alpha=1)
            # plt.axis("off")
            # plt.savefig(f"../../../../local/scratch/chuber/result/minPlus_testing/original_image{i},{j}.png", bbox_inches='tight')
            # plt.close()
            # Compute new score
            perturbed_score = affact.predict_corrrise((perturbed_image.unsqueeze(0),image_name)) # 1,40
            #print(perturbed_score.shape)
            #print(original_score.shape)
            #print(original_score[0,0].item() - perturbed_score[0,0].item())
            saliency_map[:,i, j] = original_score - perturbed_score

    print("nice until here")
    print(saliency_map.shape)
    # norm_map = saliency_map.cpu().numpy()
    # norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min() + 1e-8)
    # plt.imshow(norm_map, cmap="jet", alpha=1)
    # plt.axis("off")
    # plt.colorbar()
    # plt.savefig(f"../../../../local/scratch/chuber/result/minPlus_testing/saliency_map_{image_name}.png", bbox_inches='tight')
    # plt.close()
            
    return saliency_map

def accumulated_removal(image, width, steps, mimimal_inc, max_iteration, affact, image_name):
    C, H, W = image.shape
    saliency_map = torch.zeros((40,H, W), dtype=torch.float32, device=device)
    previous_score = affact.predict_corrrise((image.unsqueeze(0), image_name)) # 1,40    
    difference_scores = 1
    iteration = 0
    i_ = 0
    j_ = 0
    best_perturbed = None
    while(difference_scores > 0 and iteration < max_iteration):
        iteration += 1
        best_score = 1
        for i in range(0, H, steps):
            for j in range(0, W, steps):
                mask = gaussian_mask((H, W), (j, i), width).to(device)  # Note (x, y) order for center
                mask = mask.unsqueeze(0).repeat(C, 1, 1)  # Match shape (C, H, W)
                perturbed_image = image * (1 - mask)
                temp_score = affact.predict_corrrise((perturbed_image.unsqueeze(0),image_name)) # 1,40
                if temp_score < best_score:
                    best_score = temp_score
                    i_ = i
                    j_ = j
                    best_perturbed = perturbed_image
        difference_scores = previous_score - best_score  
        previous_score = best_score
        saliency_map[:,i_, j_] = difference_scores
    return saliency_map
           
            


def get_gaussian_kernel2d(kernel_size, sigma):
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= kernel.sum()
    return kernel

def smooth_and_normalize_saliency(H_bar, sigma=10):
    """
    Smooth a (224, 224) saliency map with Gaussian and normalize between 0 and 1.
    Args:
        H_bar (torch.Tensor): 2D tensor, shape (224, 224)
        sigma (float): Gaussian width
    Returns:
        torch.Tensor: Smoothed and normalized saliency map (224, 224)
    """
    H_bar = H_bar.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 224, 224)

    kernel_size = int(6 * sigma) | 1  # ensure odd
    kernel = get_gaussian_kernel2d(kernel_size, sigma).to(H_bar.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1, 1, k, k)

    # Convolve with padding to preserve shape
    D = F.conv2d(H_bar, kernel, padding=kernel_size // 2)

    # Min-max normalize
    D_min, D_max = D.min(), D.max()
    MinPlus = (D - D_min) / (D_max - D_min + 1e-8)

    return MinPlus.squeeze(0).squeeze(0) 
    

def main():
    args = command_line_options()
    
    if os.path.exists(args.output_directory):
        shutil.rmtree(args.output_directory)
    
    os.makedirs(args.output_directory, exist_ok=True)

    startTime = datetime.now()
    
    file_lists = [
        os.path.join(args.protocol_directory,
                     f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    ]


    # CelebA_dataset = attribute_cam.CelebA(file_lists,
    #                                args.source_directory,
    #                                number_of_images=args.image_count)
    
    celebA_dataset = attribute_cam.CelebA_perturb(file_lists,
                                 args.source_directory,
                                 number_of_images=args.image_count, 
                                 cam_directory = os.path.join(args.output_directory))
    
    file_list_path = os.path.join(
        args.protocol_directory,
        f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    
    
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
        image_paths = image_paths[::-1]

    N = args.masks
    width = 30 # original paper 10, bilder sind dort aber nur 112x112
    steps = 6 #original paper 30
    num_attributes = args.attributes
    first = True
    
    affact = attribute_cam.AFFACT(args.model_type, device)
    
    number_of_images = 1
    with open(f'{args.output_directory}/img_names.txt', "w") as f:
        for img_name in tqdm(image_paths[:number_of_images]):#[:number_of_images]
            f.write(f"{img_name}\n")

    
    for attribute_idx in range(0,num_attributes):        
           os.makedirs(f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}', exist_ok=True)

    # perturb images with masks and save them
    with torch.no_grad():
        for img_name in tqdm(image_paths[:number_of_images]):#[:number_of_images]
            img_path = f"{args.source_directory}/{img_name}"
    #         print(f"Processing image: {img_path}")
            image, orig_image = load_img(img_path)
            img_name_no_ext, _ = os.path.splitext(img_name)
            
            #saliency_single_removal = single_removal(image.squeeze(0).to(device), width, steps, affact, img_name_no_ext)
            print("dskfjsadlkfj")
            print(saliency_single_removal.shape)
            saliency_accumulated_removal = accumulated_removal(image.squeeze(0).to(device),0,5,width,steps,affact,img_name_no_ext)
            
            for i in range(saliency_single_removal.shape[0]):
                #attribute_saliency = saliency_single_removal[i].unsqueeze(0) 
                #print(attribute_saliency.shape)  
                #smoothed_saliency = smooth_and_normalize_saliency(attribute_saliency.squeeze(0), 10)
            
                #print(type(saliency_single_removal), type(orig_image))
                #celebA_dataset.save_perturb(smoothed_saliency.cpu(), orig_image,f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[i]}/{img_name_no_ext}.png')
                
                attribute_saliency_acc = saliency_accumulated_removal[i].unsqueeze(0) 
                #print(attribute_saliency.shape)  
                smoothed_saliency_acc = smooth_and_normalize_saliency(attribute_saliency_acc.squeeze(0), 10)
            
                #print(type(saliency_single_removal), type(orig_image))
                celebA_dataset.save_perturb(smoothed_saliency_acc.cpu(), orig_image,f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[i]}/{img_name_no_ext}_acc.png')
                
            #perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext, N)
            
            #much faster in saving saliency maps for all attributes
            
    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
