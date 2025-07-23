import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import torchvision
from utils import load_img, pearson_correlation_multi, process_saliency, process_attributes_parallel, save_masks_as_images, generate_all_saliency_maps
import attribute_cam
from CelebA.perturb_protocol.list_names import list_names

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from skimage.transform import resize
import torch
import torch.nn.functional as F
import shutil
import random
import pytorch_grad_cam
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats

import pandas as pd
import cv2
from PIL import Image
import json

    
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

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
        default="../../../../local/scratch/chuber/results/testing",
        help="Path to folder where the output should be stored")
    
    parser.add_argument('-i',
                        '--image-count',
                        type=int,
                        help="if given, limit the number of images")
    parser.add_argument('-m',
                        '--model-type',
                        default='unbalanced',
                        choices=['balanced', 'unbalanced'],
                        help="Can be balanced or unbalanced")
    parser.add_argument(
        '--gpu',
        action="store_false",
        help='Do not use GPU acceleration (will be **disabled** when selected)'
    )
    parser.add_argument(
        '-masks',
        '--masks',
        default=3000,
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
    parser.add_argument(
        '-patchsize',
        '--patchsize',
        default=30,
        type=int,
        help='Size of one patch'
    )
    
    args = parser.parse_args()

    return args


# Generate masks
def generate_masks(N,num_patches, patch_size, image_size=(224, 224)):
    
    masks =  torch.full((N, 1, image_size[0], image_size[1]), -1.0, dtype=torch.float32, device=device)
    masks_activation = torch.ones((N, 1, image_size[0], image_size[1]), dtype=torch.float32, device=device)


    for i in range(N):
        for _ in range(num_patches):
            # Random patch position
            x = random.randint(0, image_size[0] - patch_size)
            y = random.randint(0, image_size[1] - patch_size)

            patch = torch.randint(0, 2, (patch_size, patch_size), dtype=torch.float32, device=device)
            zero_patch = torch.zeros((patch_size,patch_size),dtype = torch.float32, device = device)
            # Add the patch into the mask
            masks[i, 0, y:y+patch_size, x:x+patch_size] = patch
            masks_activation[i,0,y:y+patch_size, x:x+patch_size] = zero_patch
            
    return masks, masks_activation


def generate_masks_from_loaded_coords(patch_size, image_size=(224, 224), load_path=None):
    # Load saved coordinates
    with open(load_path, 'r') as f:
        coords = json.load(f)

    N = len(coords)
    masks = torch.full((N, 1, image_size[0], image_size[1]), -1.0, dtype=torch.float32, device=device)
    masks_activation = torch.ones((N, 1, image_size[0], image_size[1]), dtype=torch.float32, device=device)

    for i, image_coords in enumerate(coords):
        for x, y in image_coords:
            patch = torch.randint(0, 2, (patch_size, patch_size), dtype=torch.float32, device=device)
            masks[i, 0, y:y+patch_size, x:x+patch_size] = patch
            masks_activation[i, 0, y:y+patch_size, x:x+patch_size] = 0.0

    return masks, masks_activation



def apply_and_save_masks(image, masks, output_dir, img_name, N=20):
    os.makedirs(output_dir, exist_ok=True)  

    if image.dim() == 4:
        image = image.squeeze(0)
    original = image.clone()
    # Save as PNG
    torchvision.io.image.write_png((original.cpu() * 255).byte(), f'{output_dir}/original_image_{img_name}.png')
    perturbed_images = []
    #image = image.to(device)
    image_expanded = image.unsqueeze(0).expand(masks.shape[0], -1, -1, -1)  # Shape: (N, 3, H, W)

    masks_expanded = masks.expand(-1, 3, -1, -1)
    perturbed = image_expanded.clone()

    # Set to black where mask == 0
    perturbed[masks_expanded == 0] = 0.0

    # Set to white where mask == 1
    perturbed[masks_expanded == 1] = 1.0
    
    # Generate filenames
    perturbed_filenames = [f'perturbed_image_{img_name}_{i}' for i in range(masks.shape[0])]
    perturbed = perturbed.cpu()
    return perturbed, perturbed_filenames


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
    num_patches = 1 # original paper 10, bilder sind dort aber nur 112x112
    patch_size = args.patchsize #original paper 30
    num_attributes = args.attributes
    first = True

    #masks, masks_activation = generate_masks(N,num_patches,patch_size)
    masks, masks_activation = generate_masks_from_loaded_coords(patch_size, image_size=(224, 224), load_path=os.path.join(args.protocol_directory,"mask_coordinates_3000masks_patch_30.txt"))
    masks = masks.cpu()
    masks_activation = masks_activation.to(device)
    save_masks_as_images(masks,f'{args.output_directory}/masken')
    save_masks_as_images(masks_activation,f'{args.output_directory}/masken_activation')
    
    affact = attribute_cam.AFFACT(args.model_type, device)
    
    number_of_images = 10
    with open(f'{args.output_directory}/img_names.txt', "w") as f:
        for img_name in tqdm(image_paths):#[:number_of_images]
            f.write(f"{img_name}\n")

    
    for attribute_idx in range(0,num_attributes):        
           os.makedirs(f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}', exist_ok=True)

    # perturb images with masks and save them
    with torch.no_grad():
        for img_name in tqdm(image_paths):#[:number_of_images]
            img_path = f"{args.source_directory}/{img_name}"
    #         print(f"Processing image: {img_path}")
            image, orig_image = load_img(img_path)
            img_name_no_ext, _ = os.path.splitext(img_name)

            perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext, N)
            if(first):
                save_masks_as_images(perturbed_images[0],f'{args.output_directory}/masks_images')
                first = False
            
            original_score = affact.predict_corrrise((image,f"original_{img_name_no_ext}")).to(device)
            scores_of_images = affact.predict_corrrise(perturbed_images) # shape (batch_size,A)        
            
            saliency_maps = generate_all_saliency_maps(masks_activation, scores_of_images, original_score, device) #shape (A,1,H,W)
            process_attributes_parallel(saliency_maps, orig_image, img_name_no_ext, num_attributes, celebA_dataset, args)

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()