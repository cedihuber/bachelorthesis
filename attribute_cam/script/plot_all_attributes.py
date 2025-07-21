import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.image as mpimg
import argparse
import attribute_cam
import os

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
        '-o',
        '--output-directory',
        default="../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch",
        help="Path to folder where the output should be stored")
    parser.add_argument(
      '-f',
      '--filters',
      nargs="+",
      default = ["majority_n", "minority_n","majority_p", "minority_p"],
      choices = list(attribute_cam.FILTERS.keys()),
      help="Average cams images with the given filters")
    args = parser.parse_args()

    return args


def plot(filter_key, args):
    attrs = Attrs
    images = [mpimg.imread(os.path.join(args.output_directory,"test-"+filter_key,attr+".png")) for attr in attrs]
   
    with PdfPages(os.path.join(args.output_directory,"a_test-"+filter_key+".pdf")) as pdf:
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
        

# def plot_as_png(filter_key):
#     attrs = Attrs
#     images = [mpimg.imread(f"../../../../local/scratch/chuber/corrRiseResultAdapted/balanced/corrRise_masks_black_2000_masks/{filter_key}/" + attr + ".png") for attr in attrs]

#     fig, axes = plt.subplots(5, 8, figsize=(24, 15))  
#     for i, ax in enumerate(axes.flat):
#         if i < len(images):
#             ax.imshow(images[i])
#             ax.axis('off')
#             ax.set_title(f"{attrs[i]}", fontsize=14)
#         else:
#             ax.axis('off')

#     plt.subplots_adjust(wspace=0.3, hspace=0.5)
#     output_path = f'../../../../local/scratch/chuber/result/corrRise_masks_blurry_10batchs_30size_500masks_1000images/a_{filter_key}.png'
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved PNG for {filter_key} at {output_path}")
 
 
def main():
    args = command_line_options()
    filter_keys = args.filters
    for filter_key in filter_keys:
            plot(filter_key, args)

if __name__ == "__main__":
    main()
    