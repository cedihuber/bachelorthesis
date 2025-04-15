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
import torch.utils.bottleneck as bottleneck
from torch.utils.data import DataLoader, Dataset

# Define Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, prefix, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = f'{self.prefix}/{self.image_paths[idx]}'
        image, _ = load_img(img_path)  # Load image using your existing function
        if self.transform:
            image = self.transform(image)
        return image

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
        default="../../../../local/scratch/chuber/result/rise",
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
    return image, image.unsqueeze(0)


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
    
        
       
            
def generate_saliency_map(masks, img_name,p, attribute_idx, scores_of_images):    
    
    attribute_scores = scores_of_images[:, attribute_idx].to(device)  # (500,)
    attribute_scores = attribute_scores.view(-1, 1, 1, 1)  # reshape to (500, 1, 1, 1)

    # weighted masks
    weighted_masks = masks * attribute_scores  # (500, 1, 224, 224)

    # sum them up
    saliency_map = torch.sum(weighted_masks, dim=0)  # (1, 224, 2245)

    # optionally normalize
    saliency_map /= (masks.shape[0] + 1e-8) * p
    return saliency_map



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


    CelebA_dataset = attribute_cam.CelebA(file_lists,
                                   args.source_directory,
                                   number_of_images=args.image_count)
    
    file_list_path = os.path.join(
        args.protocol_directory,
        f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    
    
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
        image_paths = image_paths[::-1]

    dataset = ImageDataset(image_paths,args.source_directory)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    
    N = args.masks
    s = 8 # not sure if s should be devided in height and width
    p1 = args.percentage #modifiy and check results
    num_attributes = args.attributes
    first = True

    print(f"Perturbt {len(CelebA_dataset)} images")
    masks = generate_masks(N, s, p1).to(device)
    save_masks_as_images(masks,f'{args.output_directory}/masken')
    
    affact = attribute_cam.AFFACT(args.model_type, device)

    
    for attribute_idx in range(0,num_attributes):        
           os.makedirs(f'{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}', exist_ok=True)

    # perturb images with masks and save them
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader)):
            # Move images to GPU if available
            images = images.to(device)

            for img_idx, image in enumerate(images):
                img_name = image_paths[batch_idx * 32 + img_idx]  # Get image name from paths
                img_name_no_ext, _ = os.path.splitext(img_name)

                # Apply masks and save perturbed images
                perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext, N)
                if first:
                    print(perturbed_images[0].shape)
                    save_masks_as_images(perturbed_images[0], f'{args.output_directory}/masks_images')
                    first = False

                scores_of_images = affact.predict_perturbed(perturbed_images, f'{args.output_directory}/Prediction-perturb2.csv')

                # Generate saliency map
                for attribute_idx in range(0, num_attributes):
                    saliency_map = generate_saliency_map(masks, img_name, args.percentage, attribute_idx, scores_of_images)

                    # Save saliency map as image
                    saliency = saliency_map.squeeze(0).cpu()
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

                    saliency_np = saliency.numpy()
                    colormap = plt.get_cmap("jet")
                    colored_map = colormap(saliency_np)  # shape: (224, 224, 4)

                    colored_rgb = (colored_map[:, :, :3] * 255).astype(np.uint8)

                    save_path = f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png"
                    Image.fromarray(colored_rgb).save(save_path)

                    saliency_np = saliency_map.squeeze(0).cpu().numpy()
                    np.save(f"{args.output_directory}/{attribute_cam.dataset.ATTRIBUTES[attribute_idx]}/{img_name_no_ext}.png.npy", saliency_np)

            del perturbed_images, saliency_map
            torch.cuda.empty_cache()

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()#bottleneck.run('main()')
