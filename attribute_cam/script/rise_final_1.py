import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import torch
from PIL import Image
import shutil
import torch.nn.functional as F
 
from concurrent.futures import ThreadPoolExecutor
 
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
 
def save_cam(img,mask,save_path):
    np.save(save_path+".npy", mask)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.
    cam = (1 - 0.5) * heatmap + 0.5 * img
    cam = cam / np.max(cam)
    overlay = np.uint8(255 * cam)
    torchvision.io.write_png(torch.tensor(overlay.transpose(2,0,1), dtype=torch.uint8), save_path)
 
 
header=[
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
    ]
   
def affact():
    from importlib.machinery import SourceFileLoader
    model_file = "/local/scratch/xzhang/attribute-cam/attribute_cam/AFFACT.py"
    weight_file = "/local/scratch/xzhang/attribute-cam/attribute_cam/AFFACT_unbalanced.pth"
    MainModel = SourceFileLoader("MainModel", model_file).load_module()
    network = torch.load(weight_file)
    network.identity = torch.nn.Identity()
 
    return network  
 
# Generate masks
def generate_masks(N, s, p1, input_size=(224, 224)):
    cell_size = torch.ceil(torch.tensor(input_size, dtype=torch.float32) / s)
    up_size = (s + 1) * cell_size  # Scaling factor for upsampling
 
    # Generate random binary grid masks (N, s, s) with probability p1
    grid = (torch.rand(N, s, s) < p1).float()
 
    # Random crop offsets
    x_offsets = torch.randint(0, int(cell_size[0]), (N,))
    y_offsets = torch.randint(0, int(cell_size[1]), (N,))
 
    # Upsample grid to match the input size
    grid = grid.unsqueeze(1)  # Shape: (N, 1, s, s)
    grid_upsampled = F.interpolate(grid, size=tuple(map(int, up_size)), mode="bilinear", align_corners=False)
 
    # Crop each mask based on offsets
    masks = torch.stack([
        grid_upsampled[i, :, x:x + input_size[0], y:y + input_size[1]]
        for i, (x, y) in enumerate(zip(x_offsets, y_offsets))
    ])
 
 
    return masks
 
def generate_saliency_map(masks,p, index, scores_of_images):    
    # breakpoint()
    attribute_scores = scores_of_images[:, index]  # (500,)
    #print(masks.shape)
    # weighted masks
 
    filtered_scores = attribute_scores # (N,)
    filtered_masks = masks  # (N, 1, 224, 224)
   
    filtered_scores = filtered_scores.view(-1, 1, 1, 1)  # reshape to (500, 1, 1, 1)
   
    weighted_masks = filtered_masks * filtered_scores  # (500, 1, 224, 224)
   
    saliency_map = torch.sum(weighted_masks, dim=0)  # (1, 224, 224)
 
    # optionally normalize
    saliency_map /= (filtered_masks.shape[0] * p) + 1e-8
    return saliency_map
 
 
 
N = 500
s = 8
p1 = 0.5
# masks = generate_masks(N, s, p1, input_size=(224, 224))
# # breakpoint()
# print(masks.shape)
# torch.save(masks,"mask.pt")
 
 
masks = torch.load("mask.pt")  # torch.Size([500, 1, 224, 224])
# for save
directories = ["/local/scratch/xzhang/srv/test/results/unbalanced/" + header[i] + "/" for i in range(40)]
for directory in set(directories):  # Use set to avoid creating the same directory multiple times
    os.makedirs(directory, exist_ok=True)
 
 
data = pd.read_csv("/local/scratch/xzhang/Some-Ideas/XAI-in-FAC/aligned_224x224_test_filtered_0.1.csv")
img_path = "/local/scratch/datasets/CelebA/aligned_224x224/"
model = affact()
# device = torch.device("cuda:0")
os.environ["CUDA_VISIBLE_DEVICES"]  = "2,3,4,6"
gpus = [0,1,2,3]
model.cuda()
model =  DataParallel(model, device_ids = gpus)
 
# device = torch.device("cpu")
 
# model.to(device)
model.eval()
tf = torchvision.transforms.ToTensor()
with torch.no_grad():
    for index in tqdm(range(len(data))):
        path_img = data.loc[index]["Path"]
 
        img = Image.open(img_path + path_img)
        img = tf(img)
        # breakpoint()
        perturbed_images = img * masks
        # test_img = perturbed_images[0]
        # torchvision.io.image.write_png(torch.tensor(test_img*255, dtype=torch.uint8),"test.png")
        # predictions = model(perturbed_images.cuda())
        predictions_1 = model(perturbed_images[:256].cuda()).cpu()
        predictions_2 = model(perturbed_images[256:].cuda()).cpu()
        predictions = torch.concat([predictions_1,predictions_2])
        # predictions = model(perturbed_images.to(device)).abs()
        for index in range(40):
            current_saliency_map = generate_saliency_map(masks,0.5, index, predictions)[0]
            # attr = header[index]
            save_path = directories[index] + path_img
            # overlay = pytorch_grad_cam.utils.image.show_cam_on_image(image, current_saliency_map, use_rgb=True)
            # targets = [BinaryCategoricalClassifierOutputTarget(index)]
            # activation = cam(random_tensor.unsqueeze(0).to(device), targets)[0]
   
            save_cam(img.numpy().transpose(1, 2, 0), current_saliency_map, save_path)
        del perturbed_images, predictions
        torch.cuda.empty_cache()