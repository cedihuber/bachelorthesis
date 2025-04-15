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


import os, json
from torch.autograd import Variable
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from skimage.color import label2rgb
import matplotlib.cm as cm

#from get_shifted_landmarks import get_shifted_landmarks_df
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
print(f"Using device: {device}")  # Optional: To confirm whether GPU is used        

def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Goes through the dataset and predicts the attributes")
    parser.add_argument(
        '-w',
        '--which-set',
        default="validation",
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
        default="../../../../local/scratch/chuber/result/lime",
        help="Path to folder where the output should be stored")
    parser.add_argument('-i',
                        '--image-count',
                        default=5,
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
    image = Image.open(path).convert("RGB").resize((224, 224))
    return np.array(image)  # HWC format


def preprocess_transform(pil_image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return transf(pil_image)

def lime_batch_predict(images, model):
    tensor_images = torch.stack([
        preprocess_transform(Image.fromarray(img.astype(np.uint8)))
        for img in images
    ]).to(device)
    
    with torch.no_grad():
        logits = model(tensor_images)
        probs = torch.sigmoid(logits).cpu().numpy()  # For multi-label probs

    return probs
    
        
       
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

    affact = attribute_cam.AFFACT("balanced",
                                  device)
    model = affact.model()
    model.eval()
     
    # not useful just for testing
    # Convert the NumPy array back to a PIL image
    # img_pil = Image.fromarray(image_np)
    # # Save it (choose any path you like)
    # img_pil.save(f'{args.output_directory}/testimage.png')
    
    
    explainer = lime_image.LimeImageExplainer()

    attr_idx = 1
    
    # explanation = explainer.explain_instance(image_np, 
    #                                      lime_batch_predict, # classification function
    #                                      labels=1, 
    #                                      hide_color=0, 
    #                                      num_samples=10)



    # temp, mask = explanation.get_image_and_mask(
    #     attr_idx, positive_only=True, num_features=10, hide_rest=False
    # )
 
    # img_boundaries = mark_boundaries(temp / 255.0, mask)
    # img_colored = label2rgb(mask, temp, bg_label=0, alpha=0.4)
    
    CelebA_dataset = attribute_cam.CelebA(file_lists,
                                   args.source_directory,
                                   number_of_images=args.image_count)
    
    file_list_path = os.path.join(
        args.protocol_directory,
        f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    print(image_paths[0])

    avg_mask = None
    for i in tqdm(range(args.image_count-1), desc="Processing images"):
        img_path = image_paths[i]
        image_np = load_img(f"{args.source_directory}/{img_path}")
        
        # Get LIME explanation for the current image
        explanation = explainer.explain_instance(image_np, 
                                                 lambda imgs: lime_batch_predict(imgs,model), 
                                                 labels=[1], 
                                                 hide_color=0, 
                                                 num_samples=10)
        
        # Get the explanation mask
        print(f"Available labels in explanation: {explanation.top_labels}")
        attr_idx = 1
        if(attr_idx not in explanation.top_labels):
            continue
        
        temp, mask = explanation.get_image_and_mask(
            attr_idx, positive_only=True, num_features=10, hide_rest=False
        )
        
        del explanation
        torch.cuda.empty_cache()
        
        if avg_mask is None:
            avg_mask = mask.astype(np.float32)
        else:
            avg_mask += mask.astype(np.float32)

   # Normalize the average mask (if you haven't already done it)
    avg_mask /= args.image_count

    # Convert the mask to uint8 for compatibility with mark_boundaries
    avg_mask_int = (avg_mask * 255).astype(np.uint8)
    
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img_boundaries)
    # plt.imshow(img_colored)
    # plt.title(f"LIME Explanation - Attribute {attr_idx}")
    # plt.axis('off')
    
    
    # # Save
    # fname = os.path.splitext(os.path.basename(img_path))[0]
    # plt.savefig(f"{args.output_directory}/lime.png", bbox_inches='tight')
    # plt.close()
    
    
    img_pil = Image.open(f"{args.source_directory}/{image_paths[0]}").convert("RGB")
    img_np = np.array(img_pil)

    img_colored = label2rgb(avg_mask, img_np, bg_label=0, alpha=0.4)
    img_boundaries = mark_boundaries(img_colored, avg_mask_int)

    # Plot and save the result
    plt.figure(figsize=(6, 6))
    plt.imshow(img_boundaries)
    plt.title(f"Average LIME Explanation")
    plt.axis('off')
    plt.savefig(f"{args.output_directory}/average_lime.png", bbox_inches='tight')
    plt.close()
    
    
    
    normalized_mask = (avg_mask - np.min(avg_mask)) / (np.max(avg_mask) - np.min(avg_mask) + 1e-8)

    # Get a heatmap from the mask using a colormap (e.g., 'jet', 'viridis', 'hot')
    heatmap = cm.jet(normalized_mask)[:, :, :3]  # discard alpha channel from colormap

    # Resize heatmap if needed to match image size (just in case)
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap = resize(heatmap, img_np.shape[:2], preserve_range=True)

    # Blend heatmap with the original image
    blended = (0.6 * img_np / 255.0 + 0.4 * heatmap)

    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.title("LIME Heatmap Overlay")
    plt.axis('off')
    plt.savefig(f"{args.output_directory}/average_lime_heatmap.png", bbox_inches='tight')
    plt.close()
    
    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
