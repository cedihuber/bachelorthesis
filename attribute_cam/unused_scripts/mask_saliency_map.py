import argparse
import os
from datetime import datetime
import attribute_cam
import tabulate
import numpy as np
import torch
import torchvision
import os


def mask_saliency(image_name, directory):
    os.makedirs(directory, exist_ok=True)
    masks, mask_sizes = attribute_cam.get_masks()
    image = torchvision.io.image.read_image(image_name)
    #torchvision.io.write_png(image, os.path.join(directory, "original.png"))
    for attribute,mask in masks.items():
        # write mask to file
        maskname = os.path.join(directory, attribute + "_mask.png")
        torchvision.io.write_png(torch.as_tensor([mask,mask,mask]), maskname)

        # write masked image to file
        overlay_image = np.where(mask>0,image,0)
        overlayname = os.path.join(directory, attribute + "_overlay.png")
        torchvision.io.write_png(torch.tensor(overlay_image, dtype=torch.uint8), overlayname)

mask_saliency("../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch/Bags_Under_Eyes/182655_activation.png","../../../../local/scratch/chuber/Finalresults")