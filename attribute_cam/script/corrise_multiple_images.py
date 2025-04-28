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
from dataloader import CustomImageDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

#from get_shifted_landmarks import get_shifted_landmarks_df
    
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu") # mit : cuda: 0 kann ich angeben auf welcher gpu nummer, gpustat um gpu usage zu schauen
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
        default="../../../../local/scratch/chuber/result/corrrise_multiple_images_10batchs_60size",
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

# Generate masks
def generate_masks(N,num_patches, patch_size, image_size=(224, 224)):
    
    masks = torch.zeros((N, 1, image_size[0], image_size[1]), dtype=torch.float32, device=device)

    for i in range(N):
        for _ in range(num_patches):
            # Random patch position
            x = random.randint(0, image_size[0] - patch_size)
            y = random.randint(0, image_size[1] - patch_size)

            # Random values in [0, 1] for the patch
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


def apply_and_save_masks(image, masks, output_dir, img_names, N=20):
    os.makedirs(output_dir, exist_ok=True)  
    #print(f'image{image.shape}, masks{masks.shape}')

    
    #image_expanded = image.unsqueeze(0).expand(masks.shape[0], -1, -1, -1)  # Shape: (N, 3, H, W)
    masks = masks.expand(-1, 3, -1, -1)
    # Apply masks using batch-wise multiplication
    # Masks shape: (N, 1, H, W) -> Expand to (N, 3, H, W) to match image
    #print(f'{image.unsqueeze(1).shape}{masks.unsqueeze(0).shape}')
    perturbed_images = image.unsqueeze(1) * masks.unsqueeze(0) # batch_size,500, 3, 224, 224
    perturbed_images = perturbed_images.reshape(-1, 3, 224, 224)
    # print(perturbed_images.shape)

    perturbed_filenames = []
    for img_name in img_names:
        # Generate filenames
        perturbed_filenames.extend([f'perturbed_image_{img_name}_{i}' for i in range(masks.shape[0])])
    #print(len(perturbed_images))
    #print(perturbed_images[0].shape)
    return perturbed_images, perturbed_filenames #list(zip(perturbed_images, perturbed_filenames))
    
        
def pearson_correlation_batch(scores, masks): # scores shape (500), masks shape (500,50176) 
    # pearson correlation = sum ( (x - x.mean) * (y-y. mean) ) / sq_root( sum( (x-x.mean)^2 ) * sum( (y-y.mean)^2 ) )
    
    #print(f'scores={scores.shape} masks={masks.shape}')
    scores = scores - scores.mean()
    masks = masks - masks.mean(dim=0)
    nominater = torch.matmul(scores, masks)
    
    #print(f'x = {x.shape}, y = {y.shape}')
    scores_norm = torch.norm(scores)
    masks_norm = torch.norm(masks, dim=0)
    #print(f'x_norm = {x_norm.shape}, {x_norm}, y_norm = {y_norm.shape}')

    denom = scores_norm * masks_norm
    denom[denom == 0] = 1e-8  # avoid division by zero
    #print(f'denominater{denom.shape}')
    corr = nominater / denom  # shape: (M,)
    #print(f'denominater{corr.shape}')
    return corr


def generate_saliency_map(masks, img_name, p, attribute_idx, attribute_scores, path):    
    # attribute_scores: (500,)
    # masks: (500, 1, 224, 224)
    #print(masks)
    N, _, H, W = masks.shape
    M = H * W

    masks_flat = masks.view(N, -1).squeeze(1)  # shape: (500, 224*224)
    attribute_scores = attribute_scores.to(device)
    masks_flat = masks_flat.to(device)

    #saliency_flat = pearson_correlation_batch(attribute_scores, masks_flat)  # shape: (224*224,)
    #saliency_map = saliency_flat.view(1, H, W)  # reshape to (1, 224, 224)
    
    #i think this is wrong
    saliency_flat_all = pearson_correlation_batch(attribute_scores, masks_flat) 
    saliency_maps = saliency_flat_all.view(1, H, W) 
    
    return saliency_maps



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
    patch_size = 60 #original paper 30
    p1 = args.percentage #modifiy and check results
    num_attributes = args.attributes
    first = True

    masks = generate_masks(N,num_patches,patch_size).to(device)
    save_masks_as_images(masks,f'{args.output_directory}/masken')
    
    affact = attribute_cam.AFFACT(args.model_type, device)
    
    number_of_images = 5
    with open(f'{args.output_directory}/img_names.txt', "w") as f:
        for img_name in tqdm(image_paths[:number_of_images]): #[:number_of_images]
            f.write(f"{img_name}\n")

    
    for attribute_idx in range(0,num_attributes):        
           os.makedirs(f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}', exist_ok=True)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_paths = []
    
    for img_name in tqdm(image_paths[:number_of_images]): #[:number_of_images]
            img_paths.append(f"{args.source_directory}/{img_name}")
    print(len(img_paths))

    batch_size = 1
    dataset = CustomImageDataset(image_paths=img_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Number of batches: {len(data_loader)}")
    
    # perturb images with masks and save them
    with torch.no_grad():
       for i, (image, orig_image) in enumerate(tqdm(data_loader)):
            img_names = []
            for l in range(i*batch_size,(i+1)*batch_size):
                img_names.append(image_paths[l].split(".")[0])
            #print(f'length imagenames{len(img_names)}')
            #print(img_names[0])
            #print(img_names)

            perturbed_images = apply_and_save_masks(image.to(device), masks, args.output_directory, img_names, N)
            #print(perturbed_images.shape)
            if(first):
                save_masks_as_images(perturbed_images[0][0].squeeze(0),f'{args.output_directory}/masks_images')
                first = False

            scores_of_images = affact.predict_corrrise_batches(perturbed_images) #500*batch_size,40   
            print(f'scores{scores_of_images.shape}')
    #         # Generate saliency map
            #saliency_maps = generate_all_saliency_maps(masks, scores_of_images)
            for i in range(batch_size):
                #print(f'scores_of_image {scores_of_image.shape}')
                for attribute_idx in range(num_attributes):
                    #saliency = saliency_maps[attribute_idx]
                    img_name_no_ext = img_names[i]
                    saliency = generate_saliency_map(masks, img_name, args.percentage, attribute_idx, scores_of_images[i*500:(i+1)*500,attribute_idx], f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}')
                    positive_saliency = torch.clamp(saliency.squeeze(0), min = 0).cpu()
                    saliency = saliency.squeeze(0).cpu() 
                    
                    # Normalize to [0, 1]
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8) # shape (224,224)
                    positive_saliency = (positive_saliency - positive_saliency.min()) / (positive_saliency.max() - positive_saliency.min() + 1e-8) # shape (224,224)

                    #print(f'saliency {saliency.shape}')
                    #print(f'positivesaliency {positive_saliency.shape}')
                    #print(f'original {orig_image[i].shape} {orig_image[i].max()} {orig_image[i].min()} {type(orig_image[i])}')
                    #print(f'saliency {positive_saliency.shape} {positive_saliency.max()} {positive_saliency.min()} {type(positive_saliency[0])}')
                    # NOTE: The source image for this function is float in range [0,1]
                    # the ouput of it is uint8 in range [0,255]
                    overlay = pytorch_grad_cam.utils.image.show_cam_on_image(orig_image[i].numpy().transpose(1, 2, 0), positive_saliency.numpy(), use_rgb=True)
    
                    # save CAM activation
                    celebA_dataset.save_cam(positive_saliency, overlay, attribute_cam.dataset.ATTRIBUTES[attribute_idx], img_name_no_ext) #attribute ist namen von attribute
                

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
