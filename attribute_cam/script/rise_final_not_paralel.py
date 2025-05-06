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
from concurrent.futures import ThreadPoolExecutor
import pytorch_grad_cam

import torch
import cv2
import pandas as pd
import numpy as np
import math
import random
from PIL import Image
import torchvision
import scipy
import os
from torch.nn import DataParallel
from tqdm import tqdm
import pdb



def save_saliency(colored_rgb, saliency_np, base_path):
    # Save image
    Image.fromarray(colored_rgb).save(f"{base_path}.png")
    # Save numpy array
    np.save(f"{base_path}.png.npy", saliency_np)

#from get_shifted_landmarks import get_shifted_landmarks_df
    
device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu") # mit : cuda: 0 kann ich angeben auf welcher gpu nummer, gpustat um gpu usage zu schauen
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
        default="../../../../local/scratch/chuber/result/new_rise_implementation_500masks_balanced_logits_new_saving",
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
        default=500,
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


# Generate masks
def generate_masks(N, s, p1, input_size=(224, 224)):
    cell_size = torch.ceil(torch.tensor(input_size, dtype=torch.float32, device = device) / s) 
    up_size = (s + 1) * cell_size  # Scaling factor for upsampling

    # Generate random binary grid masks (N, s, s) with probability p1
    grid = (torch.rand(N, s, s, device=device) < p1).float()

    # Random crop offsets
    x_offsets = torch.randint(0, int(cell_size[0]), (N,), device=device)
    y_offsets = torch.randint(0, int(cell_size[1]), (N,), device=device)

    # Upsample grid to match the input size
    grid = grid.unsqueeze(1)  # Shape: (N, 1, s, s)
    grid_upsampled = F.interpolate(grid, size=tuple(map(int, up_size)), mode="bilinear", align_corners=False)

    # Crop each mask based on offsets
    masks = torch.stack([
        grid_upsampled[i, :, x:x + input_size[0], y:y + input_size[1]]
        for i, (x, y) in enumerate(zip(x_offsets, y_offsets))
    ])


    return masks


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


def apply_and_save_masks(image, masks, output_dir, img_name, N=20):
    os.makedirs(output_dir, exist_ok=True)  

    if image.dim() == 4:
        image = image.squeeze(0)
    original = image.clone()
    # Save as PNG
    torchvision.io.image.write_png((original.cpu() * 255).byte(), f'{output_dir}/original_image_{img_name}.png')
    perturbed_images = []
    image = image.to(device)
    image_expanded = image.unsqueeze(0).expand(masks.shape[0], -1, -1, -1)  # Shape: (N, 3, H, W)

    # Apply masks using batch-wise multiplication
    # Masks shape: (N, 1, H, W) -> Expand to (N, 3, H, W) to match image
    perturbed_images = image_expanded * masks.expand(-1, 3, -1, -1)

    # Generate filenames
    perturbed_filenames = [f'perturbed_image_{img_name}_{i}' for i in range(masks.shape[0])]

    return perturbed_images, perturbed_filenames #list(zip(perturbed_images, perturbed_filenames))
    
        
       
            
def generate_saliency_map(masks, img_name,p, attribute_idx, scores_of_images,path):    
    
    attribute_scores = scores_of_images[:, attribute_idx].to(device)  # (500,)
    #print(masks.shape)
    # weighted masks

    filtered_scores = attribute_scores # (N,)
    filtered_masks = masks  # (N, 1, 224, 224)
    
    filtered_scores = filtered_scores.view(-1, 1, 1, 1)  # reshape to (500, 1, 1, 1)
    
    weighted_masks = filtered_masks * filtered_scores  # (500, 1, 224, 224)
    
    # scores = attribute_scores.squeeze().cpu().numpy()  # (500,)
   
    # for i, (wm, score) in enumerate(zip(weighted_masks, scores)):
    #     wm_np = wm.squeeze().cpu().numpy()

    #     plt.imshow(wm_np, cmap="jet", alpha=0.9)
    #     plt.title(f"Score: {score:.4f}, max: {wm_np.max()}, min: {wm_np.min()}", fontsize=10)
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f'{path}{i}.png', bbox_inches="tight", pad_inches=0)
    #     plt.close()
        
    # sum them up
    saliency_map = torch.sum(weighted_masks, dim=0)  # (1, 224, 224)

    # optionally normalize
    saliency_map /= (filtered_masks.shape[0] * p) + 1e-8
    return saliency_map


def main():
    executor = ThreadPoolExecutor(max_workers=8)  # Adjust depending on your CPU
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
    s = 8 # not sure if s should be devided in height and width
    p1 = args.percentage #modifiy and check results
    num_attributes = args.attributes
    first = True

    masks = generate_masks(N, s, p1).to(device)
    save_masks_as_images(masks,f'{args.output_directory}/masken')
    
    affact = attribute_cam.AFFACT(args.model_type, device)


    # os.environ["CUDA_VISIBLE_DEVICES"]  = "0,4"
    # gpus = [0,1]
    # #affact.cuda()
    # affact =  DataParallel(affact, device_ids = gpus)


    # affact.eval()
    for attribute_idx in range(0,num_attributes):        
           os.makedirs(f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}', exist_ok=True)


    number_of_images = 20
    with open(f'{args.output_directory}/img_names.txt', "w") as f:
        for img_name in tqdm(image_paths): #[:number_of_images]
            f.write(f"{img_name}\n")


    # perturb images with masks and save them
    with torch.no_grad():
        for img_name in tqdm(image_paths): #[:number_of_images]
            img_path = f"{args.source_directory}/{img_name}"
            print(f"Processing image: {img_path}")
            image, orig_image  = load_img(img_path)
            img_name_no_ext, _ = os.path.splitext(img_name)

            perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext,
                             N)
            # if(first):
            #     print(perturbed_images[0].shape)
            #     save_masks_as_images(perturbed_images[0],f'{args.output_directory}/masks_images')
            #     first = False
            scores_of_images = affact.predict_logit(perturbed_images)
            # scores_of_images_1 = affact.predict_logit(perturbed_images[:250])
            # scores_of_images_2 = affact.predict_logit(perturbed_images[250:])   
            # scores_of_images = torch.concat([scores_of_images_1,scores_of_images_2])
            
            # Generate saliency map
            for attribute_idx in range(0,num_attributes):
            
                saliency_map = generate_saliency_map(masks, img_name, args.percentage, attribute_idx, scores_of_images,f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}')
                #saliency_map 1x224x224
                #print(saliency_map.shape)
                # plt.imshow(saliency_map.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.7)
                # plt.axis("off")
                # plt.savefig(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png", bbox_inches='tight')
                # plt.close()
                saliency = saliency_map.squeeze(0).cpu() 
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                # Normalize to [0, 1]
                
            # NOTE: The source image for this function is float in range [0,1]
            # the ouput of it is uint8 in range [0,255]
                #print(f"original {orig_image.shape}, {orig_image.max()} { orig_image.min()} {type(orig_image)}")
                #print(f'saliency {saliency.shape} {saliency.max()} {saliency.min()}')
                
                celebA_dataset.save_perturb(saliency,orig_image,f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png')
                #overlay = pytorch_grad_cam.utils.image.show_cam_on_image(orig_image, saliency.numpy(), use_rgb=True)
                #celebA_dataset.save_cam(saliency, overlay, attribute_cam.dataset.ATTRIBUTES[attribute_idx], img_name_no_ext) #attribute ist namen von attribute
            
            
            
                # # Convert to numpy and apply colormap
                # saliency_np = saliency.numpy()
                # colormap = plt.get_cmap("jet")
                # colored_map = colormap(saliency_np)  # shape: (224, 224,4)
                # print(colored_map.shape)
                # # Drop alpha channel and convert to uint8 RGB
                # colored_rgb = (colored_map[:, :, :3] * 255).astype(np.uint8) #shape 224,224,3
                # print(colored_rgb.shape)
                # # Inside your per-image/attribute loop:
                # base_path = f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}"
                # #saliency_np = saliency_map.squeeze(0).cpu().numpy()  # shape: (224, 224)

                # executor.submit(save_saliency, colored_rgb, saliency_np, base_path)
                
                # # Save as image
                # save_path = f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png"
                # Image.fromarray(colored_rgb).save(save_path)
                
                # saliency_np = saliency_map.squeeze(0).cpu().numpy()  # shape: (H, W)
                # np.save(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png.npy", saliency_np)
                
                # plt.imshow(saliency_map.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.9)
                # plt.axis("off")
                # plt.tight_layout()
                # plt.savefig(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/saliency_map_old_way{img_name_no_ext}.png", bbox_inches='tight')
                # plt.close()

            del perturbed_images, saliency_map
            torch.cuda.empty_cache()

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()