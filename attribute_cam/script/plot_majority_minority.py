import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.image as mpimg
import argparse
import attribute_cam
import os
import json

Attrs=[
    'Attractive',
    'Mouth_Slightly_Open',
    'Smiling',
    'Wearing_Lipstick',
    'High_Cheekbones',
    'Male',
    'Heavy_Makeup',
    'Wavy_Hair',
    'Oval_Face',
    'Pointy_Nose',
    'Arched_Eyebrows',
    'Big_Lips',
    'Black_Hair',
    'Big_Nose',
    'Young',
    'Straight_Hair',
    'Bags_Under_Eyes',
    'Brown_Hair',
    'Wearing_Earrings',
    'No_Beard',
    'Bangs',
    'Blond_Hair',
    'Bushy_Eyebrows',
    'Wearing_Necklace',
    'Narrow_Eyes',
    '5_o_Clock_Shadow',
    'Receding_Hairline',
    'Wearing_Necktie',
    'Rosy_Cheeks',
    'Eyeglasses',
    'Goatee',
    'Chubby',
    'Sideburns',    
    'Blurry',
    'Wearing_Hat',
    'Double_Chin',
    'Pale_Skin',
    'Gray_Hair',
    'Mustache',
    'Bald'
    ]


def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Goes through the dataset and predicts the attributes")
    parser.add_argument(
        '-s',
        '--source-directory',
        default='/local/scratch/datasets/CelebA/aligned_224x224',
        help="Select directory containing the input dataset")
    parser.add_argument(
      '-p', '--protocol-directory',
      default='CelebA/protocol',
      help="Select directory containing the original filelists defining the protocol and ground truth of CelebA"
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        default="../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch",
        help="Path to folder where the output should be stored")
    parser.add_argument(
      '-f',
      '--filters',
      nargs="+",
      default = ["pr=1","pr=-1"],
      choices = list(attribute_cam.FILTERS.keys()),
      help="Average cams images with the given filters")
    args = parser.parse_args()

    return args


def plot(filter_key, args, majority):
    file_path = os.path.join(args.protocol_directory, "majority.txt")
    with open(file_path, 'r') as f:
        loaded_dict = json.load(f)
    attrs = []    
    if(filter_key == "pr=1"):
        if majority:
            for attr in Attrs:
                if(loaded_dict[attr] == 1):
                    attrs.append(attr)
        else:
            for attr in Attrs:
                if(loaded_dict[attr] == -1):
                        attrs.append(attr)
    if(filter_key == "pr=-1"):
        if majority:
            for attr in Attrs:
                if(loaded_dict[attr] == -1):
                    attrs.append(attr)
        else:
            for attr in Attrs:
                if(loaded_dict[attr] == 1):
                        attrs.append(attr)
            
    if majority:
        group = "majority"
    else:
        group = "minority"
    images = [mpimg.imread(os.path.join(args.output_directory,"test-"+filter_key,attr+".png")) for attr in attrs] 
    with PdfPages(os.path.join(args.output_directory,"a_test-"+filter_key+group+".pdf")) as pdf:
        fig, axes = plt.subplots(5, 8, figsize=(24, 15))  
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i])  
                ax.axis('off')  
                ax.set_title(f"{attrs[i]}", fontsize=14)
            else:
                ax.axis('off')  
 
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        pdf.savefig(fig)
        plt.close(fig)
        
        
def main():
    args = command_line_options()
    filter_keys = args.filters
    for filter_key in filter_keys:
            plot(filter_key, args, True)
            plot(filter_key, args, False)

if __name__ == "__main__":
    main()
    