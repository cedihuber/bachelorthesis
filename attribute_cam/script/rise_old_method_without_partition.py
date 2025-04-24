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
        default="../../../../local/scratch/chuber/result/rise_old_method2",
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
        default=100,
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
    image = image / 255.
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


def save_masks_as_images(masks, output_dir="masks_output"):
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
    
        
       
            
def generate_saliency_map(perturbed_images_names, masks, N, p, attribute_idx, scores_dict):    

    saliency_map = torch.zeros((1, 224, 224), dtype=torch.float32, device=device)
    # Extract valid scores
    valid_scores = [scores_dict[name][attribute_idx] for name in perturbed_images_names if name in scores_dict]
    valid_scores = torch.tensor(valid_scores, dtype=torch.float32, device=device)  # Shape: (N,)
    
    # Extract corresponding masks
    valid_masks = torch.stack([masks[i] for i, name in enumerate(perturbed_images_names) if name in scores_dict])  # Shape: (N, 1, 224, 224)

    valid_scores = valid_scores.view(-1, 1, 1, 1)  # Shape: (N, 1, 1, 1)
    # Compute weighted sum of masks
    
    saliency_map = torch.sum(valid_masks * (valid_scores), dim=0)  # Shape: (1, 224, 224)
    saliency_map /=  max(valid_masks.shape[0],1) * p
    return saliency_map


def main():
    args = command_line_options()
    
    os.makedirs(args.output_directory, exist_ok=True)
    
    #delete all images in the folder that existed before not needed at the end
    for filename in os.listdir(args.output_directory):
        file_path = os.path.join(args.output_directory, filename)

        if os.path.isfile(file_path):
            os.remove(file_path) 
            print(f"Deleted file: {filename}")

    
 
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

    ground_truth_file = os.path.join(args.protocol_directory, "list_attr_celeba.txt")
    ground_truth = attribute_cam.read_list(ground_truth_file, " ", 2)
    N = args.masks
    s = 8 # not sure if s should be devided in height and width
    p1 = args.percentage #modifiy and check results
    num_attributes = 40
    count = 0

    print(f"Perturbt {len(CelebA_dataset)} images")
    masks = generate_masks(N, s, p1).to(device)
    #save_masks_as_images(masks,"result/myresult")
    perturbed_images = []
    avg_saliency_maps_positiv = {i: torch.zeros((1, 224, 224), dtype=torch.float32, device=device) for i in range(num_attributes)}
    avg_background = torch.zeros((3, 224, 224), dtype=torch.float32, device = device)
    # perturb images with masks and save them
    background = torch.zeros((3, 224, 224), dtype=torch.float32, device = device)
    count_positiv = np.zeros(num_attributes)
    
    for img_name in tqdm(image_paths):
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
        img_path = f"{args.source_directory}/{img_name}"
        print(f"Processing image: {img_path}")
        background_image, image = load_img(img_path)
        img_name_no_ext, _ = os.path.splitext(img_name)
        with torch.no_grad():
            perturbed_images = apply_and_save_masks(image, masks, args.output_directory, img_name_no_ext,
                             N)

            affact = attribute_cam.AFFACT(args.model_type,
                                  device)
        #print(perturbed_images[0].shape)
            affact.predict_file_logit(perturbed_images,f'{args.output_directory}/Prediction-perturb2.csv')

        background_image = background_image.to(device)
        background = background_image
        avg_background += background_image
        count += 1
        scores_dict = {}  # Store image names and their corresponding scores
        with open(f"{args.output_directory}/Prediction-perturb2.csv", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                image_name = row[0]  # First column contains the perturbed image name
                scores = np.array(row[1:], dtype=np.float32)
                scores_dict[image_name] = scores
            
        # Generate saliency map
        print("generating saliency maps")
        for attribute_idx in range(num_attributes):
            perturbed_images_names = [f"perturbed_image_{img_name_no_ext.split('_')[-1]}_{i}" for i in range(0, N)] #0, N should be some parameter 
        
            saliency_map = generate_saliency_map(perturbed_images_names, masks, N, p1, attribute_idx, scores_dict)
            
            # plt.imshow(background_image.mean(dim=0).cpu().numpy())
            # plt.imshow(saliency_map.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.7)
            # plt.axis("off")
            # plt.colorbar()
            # plt.savefig(f"{args.output_directory}/saliency_map_positiv_image{img_name_no_ext}_attribute{attribute_idx}.png", bbox_inches='tight')
            # plt.close()
        
            avg_saliency_maps_positiv[attribute_idx] += saliency_map
        
        #del perturbed_images, saliency_map_positiv, saliency_map_negativ, scores_dict
        #torch.cuda.empty_cache()
        
    avg_background /= count
    
    for attribute_idx in range(num_attributes):
        avg_saliency_maps_positiv[attribute_idx] = (avg_saliency_maps_positiv[attribute_idx] -  torch.min(avg_saliency_maps_positiv[attribute_idx])) / (torch.max(avg_saliency_maps_positiv[attribute_idx]) + 1e-10)
  
        #positiv
        #plt.imshow(avg_background.mean(dim=0).cpu().numpy())
        plt.imshow(background.mean(dim=0).cpu().numpy())
        plt.imshow(avg_saliency_maps_positiv[attribute_idx].squeeze(0).cpu().numpy(), cmap="jet", alpha=0.7)
        plt.axis("off")
        plt.colorbar()
        plt.savefig(f"{args.output_directory}/saliency_map_averaged_positiv_attribute{attribute_idx}.png", bbox_inches='tight')
        plt.close()
        

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
