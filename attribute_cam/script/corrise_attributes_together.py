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

#from get_shifted_landmarks import get_shifted_landmarks_df
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # mit : cuda: 0 kann ich angeben auf welcher gpu nummer, gpustat um gpu usage zu schauen
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
        default="../../../../local/scratch/chuber/result/corrrise_20batchs_30size_1000masks_attributes_together_all_images",
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


def load_img(path, input_size=(224, 224)):
    image = torchvision.io.image.read_image(path)
    # convert to the required data type
    image = image / 255.0
    # add the required batch dimension
    image = image.unsqueeze(0)
    
    return image, image[0].numpy().transpose(1,2,0) 


# Generate masks
def generate_masks(N,num_patches, patch_size, image_size=(224, 224)):
    
    masks = torch.ones((N, 1, image_size[0], image_size[1]), dtype=torch.float32, device=device)

    for i in range(N):
        for _ in range(num_patches):
            # Random patch position
            x = random.randint(0, image_size[0] - patch_size)
            y = random.randint(0, image_size[1] - patch_size)

            # Random values in [0, 1] for the patch nur 1 oder 0 nicht zwischen 1 und 0
            patch = torch.rand((patch_size, patch_size), device=device)

            # Add the patch into the mask
            masks[i, 0, y:y+patch_size, x:x+patch_size] = patch
            
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
    # Masks shape: (N, 1, H, W) -> Expand to (N, 3, H, W) to match image # die werte nicht multiplizieren aber ersetzten, 
    perturbed_images = image_expanded * masks.expand(-1, 3, -1, -1)

    # Generate filenames
    perturbed_filenames = [f'perturbed_image_{img_name}_{i}' for i in range(masks.shape[0])]

    return perturbed_images, perturbed_filenames #list(zip(perturbed_images, perturbed_filenames))
    


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


def generate_all_saliency_maps(masks, attribute_scores):
    
    #Generate saliency maps for all attributes. Returns tensor of shape (A, 1, H, W)

    N, _, H, W = masks.shape
    M = H * W

    masks_flat = masks.view(N, -1)  # shape: (N, H*W)
    attribute_scores = attribute_scores.to(device)
    masks_flat = masks_flat.to(device)

    saliency_flat_all = pearson_correlation_multi(attribute_scores, masks_flat)  # (A, H*W)
    saliency_maps = saliency_flat_all.view(attribute_scores.shape[1], 1, H, W)  # (A, 1, H, W)
    return saliency_maps





def process_saliency(attribute_idx, saliency_maps, orig_image, img_name_no_ext, attribute_name, celebA_dataset):
    saliency = saliency_maps[attribute_idx]
    positive_saliency = torch.clamp(saliency.squeeze(0), min=0).cpu()
    saliency = saliency.squeeze(0).cpu()

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    positive_saliency = (positive_saliency - positive_saliency.min()) / (positive_saliency.max() - positive_saliency.min() + 1e-8)

    # Generate RISE overlay
    overlay = pytorch_grad_cam.utils.image.show_cam_on_image(orig_image, positive_saliency.numpy(), use_rgb=True)

    # Save RISE activation
    celebA_dataset.save_cam(positive_saliency, overlay, attribute_name, img_name_no_ext)

def process_attributes_parallel(saliency_maps, orig_image, img_name_no_ext, num_attributes, celebA_dataset):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for attribute_idx in range(num_attributes):
            attribute_name = attribute_cam.dataset.ATTRIBUTES[attribute_idx]
            futures.append(executor.submit(process_saliency, attribute_idx, saliency_maps, orig_image, img_name_no_ext, attribute_name, celebA_dataset))
        
        # Wait for all tasks to finish
        for future in futures:
            future.result()




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
    
    celebA_dataset = attribute_cam.CelebA(file_lists,
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
    num_patches = 10 # original paper 10, bilder sind dort aber nur 112x112
    patch_size = 30 #original paper 30
    p1 = args.percentage #modifiy and check results
    num_attributes = args.attributes
    first = True

    masks = generate_masks(N,num_patches,patch_size).to(device)
    save_masks_as_images(masks,f'{args.output_directory}/masken')
    
    affact = attribute_cam.AFFACT(args.model_type, device)
    
    number_of_images = 1000
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

            scores_of_images = affact.predict_corrrise(perturbed_images) # 500,40        
    #         # Generate saliency map
            saliency_maps = generate_all_saliency_maps(masks, scores_of_images) #shape (40,1,224,224)
            print(saliency_maps.shape)
            
            #much faster in saving saliency maps for all attributes
            process_attributes_parallel(saliency_maps, orig_image, img_name_no_ext, num_attributes, celebA_dataset)

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
    # # Profile your main function or any function
    # profiler = cProfile.Profile()
    # profiler.enable()

    # main()  # Replace with your main function or the function you want to profile

    # profiler.disable()

    # # Save the profile output to a file
    # profiler.dump_stats('profile_output.prof')

    # # Read the profile output from the file
    # stats = pstats.Stats('profile_output.prof')

    # # Sort by cumulative time ('cumtime') to identify the slowest parts
    # stats.sort_stats('cumtime')

    # # Print the top 20 slowest functions
    # stats.print_stats(20)