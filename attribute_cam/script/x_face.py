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
    
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # mit : cuda: 0 kann ich angeben auf welcher gpu nummer, gpustat um gpu usage zu schauen
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
        default="../../../../local/scratch/chuber/result/xFace",
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

def systematic_occlusion(image, patch_size, shape, smooth, stride):
    
    C, H, W = image.shape
    xs = ys = torch.arange(0, H - patch_size, stride, device = device)
    w = h = patch_size
    image_o, patch_masks = [], []
    for y in ys:
        for x in xs:
            mask = torch.ones((1, H, W), device=device)
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

            
            if shape == "rectangle":
                mask[:, y:y+patch_size, x:x+patch_size] = 0.0

            elif shape == "circle":
                center_y = y + patch_size // 2
                center_x = x + patch_size // 2
                dist = (yy - center_y) ** 2 + (xx - center_x) ** 2
                mask[:, dist <= (patch_size // 2) ** 2] = 0.0

            else:
                raise ValueError("Shape must be 'rectangle' or 'circle'.")
            
            # if smooth:
            #     mask = cv2.GaussianBlur(mask, (patch_size % 2 - 1, patch_size % 2 - 1), patch_size // 7)
            # fill_img = np.ones_like(img1) * (
            #     np.random.random(img1.shape) if self.fill_value == "random" else self.fill_value
            # )
            fill_img = torch.rand_like(image)
            image_o.append(image * mask + fill_img * (1 - mask))
            patch_masks.append(mask)
    return (
        torch.stack(image_o, dim=0),  # (N, C, H, W)
        torch.stack(patch_masks, dim=0),  # (N, C, H, W)
    )


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
    num_patches = 1 # original paper 10, bilder sind dort aber nur 112x112
    patch_size = 30 #original paper 30
    p1 = args.percentage #modifiy and check results
    num_attributes = args.attributes
    first = True

    #masks = generate_masks(N,num_patches,patch_size)
    #masks = masks.to(device)
    #save_masks_as_images(masks,f'{args.output_directory}/masken')
    
    affact = attribute_cam.AFFACT(args.model_type, device)
    
    number_of_images = 10
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
            image, orig_image = load_img(img_path) # image shape 1,3,224,224
            img_name_no_ext, _ = os.path.splitext(img_name)

            #perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext, N)
            perturbed_images, masks = systematic_occlusion(image.squeeze(0).to(device), 20, "rectangle", False, 20)
            perturbed_filenames = [f'perturbed_image_{img_name}_{i}' for i in range(masks.shape[0])]
            #print(perturbed_images.shape) 121,3,224,224
            #print(masks.shape) 121,1,224,224
            if(first):
                save_masks_as_images(perturbed_images,f'{args.output_directory}/masks_images')
                first = False
            original_score = affact.predict_corrrise((image,f"original_{img_name_no_ext}"))
            scores_of_images = affact.predict_corrrise((perturbed_images, perturbed_filenames)) # 500,40
            #print(original_score.shape) 1,40
            #print(scores_of_images.shape) 121,40  
            deviation = scores_of_images - original_score
            print(deviation.shape)
            for attr_idx in range(1,40):
                print("Deviation shape:", deviation.shape)
                dev_attr = deviation[:, attr_idx]  # Shape: (N,)
                # Expand to (N, H, W)
                #print(masks.shape)
                a = dev_attr[:, None, None].to(device)
                a = a.view(121, 1, 1, 1)
                b = (1.0 - masks.to(device))
                print(f"a{a.shape}b{b.shape}a*b{(b*a).shape}")
                dev_weighted = b*a
                print(f"dev_weighted{dev_weighted.shape}")
                # Average and normalize per patch area
                exp_map = dev_weighted.mean(dim=0)
                # Apply smoothing
                saliency_map = cv2.GaussianBlur(exp_map.cpu().numpy().astype(np.float32), (21, 21), 20)
                print(saliency_map.shape)
                print(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attr_idx]}/{img_name_no_ext}.png")
                plt.imshow(saliency_map.squeeze(0), cmap="jet", alpha=0.7)
                plt.axis("off")
                plt.savefig(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attr_idx]}/{img_name_no_ext}.png", bbox_inches='tight')
                plt.close()
                
            # weighted_masks = (1.0 - masks[:, :, :, 0]) * deviation[:, None, None]
            # exp_map = np.mean(weighted_masks / (patch_size ** 2), axis=0)
            # saliency_map = 
    # #         # Generate saliency map
    #         saliency_maps = generate_all_saliency_maps(masks, scores_of_images) #shape (40,1,224,224)
    #         print(saliency_maps.shape)
            
    #         #much faster in saving saliency maps for all attributes
    #         process_attributes_parallel(saliency_maps, orig_image, img_name_no_ext, num_attributes, celebA_dataset, args)

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
