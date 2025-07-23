import torchvision
import torch
import pytorch_grad_cam
from concurrent.futures import ThreadPoolExecutor
import attribute_cam
import numpy as np
import os
from PIL import Image

def load_img(path, input_size=(224, 224)):
    image = torchvision.io.image.read_image(path)
    # convert to the required data type
    image = image / 255.0
    # add the required batch dimension
    image = image.unsqueeze(0)
    
    return image, image[0].numpy().transpose(1,2,0)


def pearson_correlation_multi(x, y, original_score): # x shape (500,40) y shape (500,H*W)
    # pearson correlation = sum ( (x - x.mean) * (y-y. mean) ) / sq_root( sum( (x-x.mean)^2 ) * sum( (y-y.mean)^2 ) )

    x = x - original_score # (N, A) hier wird im gegensatz zu original pearson correlation der original score abgezogen um zu sehen ob durch das abdecken score höher oder tiefer wird
    y = y - y.mean(dim=0, keepdim=True)  # (N, M)
    nominater = torch.matmul(x.T, y)
    #print(f'x = {x.shape}, y = {y.shape}')
    x_norm = torch.norm(x, dim=0, keepdim=True)  # (1, A)
    y_norm = torch.norm(y, dim=0, keepdim=True)  # (1, H*W)
    
    denom = torch.matmul(x_norm.T, y_norm)  # (A, M)
    denom[denom == 0] = 1e-8
    
    corr = nominater / denom  # (A, M)
    sign_adjustment = torch.where(original_score.view(-1, 1) >= 0, 1.0, -1.0)  # (A, 1)
    corr = corr * sign_adjustment  # (A, M)
    
    return corr


def process_saliency(attribute_idx, saliency_maps, orig_image, img_name_no_ext, attribute_name, celebA_dataset, args):
    saliency = saliency_maps[attribute_idx]
    positive_saliency = torch.clamp(saliency.squeeze(0), min=0).cpu()
    # Normalize to [0, 1]
    positive_saliency = (positive_saliency - positive_saliency.min()) / (positive_saliency.max() - positive_saliency.min() + 1e-8)
    
    # Generate overlay
    overlay = pytorch_grad_cam.utils.image.show_cam_on_image(orig_image, positive_saliency.numpy(), use_rgb=True)
    celebA_dataset.save_cam(positive_saliency, overlay, attribute_name, img_name_no_ext)
    

def process_attributes_parallel(saliency_maps, orig_image, img_name_no_ext, num_attributes, celebA_dataset, args):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for attribute_idx in range(num_attributes):
            attribute_name = attribute_cam.dataset.ATTRIBUTES[attribute_idx]
            futures.append(executor.submit(process_saliency, attribute_idx, saliency_maps, orig_image, img_name_no_ext, attribute_name, celebA_dataset, args))
        
        # Wait for all tasks to finish
        for future in futures:
            future.result()


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

def generate_all_saliency_maps(masks, attribute_scores, original_score, device):
    
    #Generate saliency maps for all attributes. Returns tensor of shape (A, 1, H, W)

    N, _, H, W = masks.shape
    M = H * W

    masks_flat = masks.view(N, -1)  # shape: (N, H*W)
    attribute_scores = attribute_scores.to(device)
    masks_flat = masks_flat.to(device)

    saliency_flat_all = pearson_correlation_multi(attribute_scores, masks_flat, original_score)  # (A, H*W)
    saliency_maps = saliency_flat_all.view(attribute_scores.shape[1], 1, H, W)  # (A, 1, H, W)
    return saliency_maps