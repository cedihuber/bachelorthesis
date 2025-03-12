import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam

import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from skimage.transform import resize

#from get_shifted_landmarks import get_shifted_landmarks_df


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
        default="./result",
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
    args = parser.parse_args()

    return args


def load_img(path, input_size=(224, 224)):
    img = image.load_img(path, target_size=input_size)
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)  I think I dont need this
    return img, x


# Generate masks
def generate_masks(N, s, p1, input_size=(224, 224)):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        masks[i, :, :] = resize(grid[i],
                                up_size,
                                order=1,
                                mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0],
                                                     y:y + input_size[1]]

    masks = masks.reshape(-1, *input_size, 1)
    return masks


# Apply masks and save perturbed images
def perturb_and_save_images(image_path, save_dir, N=2000, s=8, p1=0.5):
    img, x = load_img(image_path)
    masks = generate_masks(N, s, p1)

    # Apply masks to image
    perturbed_images = x * masks

    for i in range(N):
        perturbed_img = perturbed_images[i].astype(np.uint8)
        plt.imsave(f"{save_dir}/perturbed_{i}.png", perturbed_img)


def apply_and_save_masks(img, x, masks, output_dir, img_name, N=2000):
    os.makedirs(output_dir,
                exist_ok=True)  # Create output directory if it doesn't exist

    original_img_path = os.path.join(output_dir,
                                     f'original_image_{img_name}.png')

    original_img = image.array_to_img(
        x.squeeze())  # Convert back to image and remove batch dimension
    original_img.save(original_img_path)

    for i in tqdm(range(N), desc="Applying masks"):
        perturbed_image = x * masks[i]  # Apply the mask to the image
        perturbed_image = perturbed_image.squeeze()  # Remove batch dimension

        # Save the perturbed image
        perturbed_img_path = os.path.join(
            output_dir, f"perturbed_image_{img_name}_{i+1}.png")

        perturbed_img = image.array_to_img(
            perturbed_image)  # Convert back to image
        perturbed_img.save(perturbed_img_path)
        
        
# def apply_and_save_mask(img, x, masks, output_dir, img_name, N=2000):
#     os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
#     original_img_path = os.path.join(output_dir, f'original_image_{img_name}.png')
#     original_img = image.array_to_img(x.squeeze())  # Convert back to image
#     original_img.save(original_img_path)
#     for i in tqdm(range(N), desc="Applying masks"):
#         perturbed_image = x * masks[i]  # Apply the mask
#         perturbed_image = perturbed_image.squeeze()  # Remove batch dimension
#         # Save the perturbed image
#         perturbed_img_path = os.path.join(output_dir, f"perturbed_image_{img_name}_{i+1}.png")
#         perturbed_img = image.array_to_img(perturbed_image)
#         perturbed_img.save(perturbed_img_path)
        
            
def generate_saliency_map(img_name, x, masks):    

    saliency_map = np.zeros_like(x[..., 0], dtype=np.float32)  # Initialize map

    #print(f"Computing saliency map for class {class_idx}")

    scores_dict = {}  # Store image names and their corresponding scores

    with open("result/myresult/Prediction-perturb.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0]  # First column contains the perturbed image name
            scores = np.array(row[1:], dtype=np.float32)  # Convert scores to float
            scores_dict[image_name] = scores  # Store in dictionary
    #print(scores_dict)
    print(img_name)
    score = scores_dict[img_name][0]
    print(scores_dict[img_name][0])
    for i in tqdm(range(masks.shape[0]), desc="Evaluating masked images"):
        # Accumulate weighted masks
        saliency_map += masks[i, :, :, 0] * score
    # Normalize the saliency map
    saliency_map /= masks.shape[0]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    return saliency_map



def main():
    args = command_line_options()
    os.makedirs("result/myresult", exist_ok=True)
    # create dataset

    startTime = datetime.now()

    file_lists = [
        os.path.join(args.protocol_directory,
                     f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    ]

    dataset = attribute_cam.CelebA(file_lists,
                                   args.source_directory,
                                   number_of_images=args.image_count)
    file_list_path = os.path.join(
        args.protocol_directory,
        f"aligned_224x224_{args.which_set}_filtered_0.1.txt")
    with open(file_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    N = 2
    s = 8
    p1 = 0.5

    print(f"Perturbt {len(dataset)} images")

    for img_name in image_paths[:args.image_count]:
        img_path = f"{args.source_directory}/{img_name}"
        print(f"Processing image: {img_path}")
        img, x = load_img(img_path)
        masks = generate_masks(N, s, p1)
        print(f"Perturbing {img_path}")
        img_name_no_ext, _ = os.path.splitext(img_name)
        apply_and_save_masks(img, x, masks, "result/myresult", img_name_no_ext,
                             N)

        #create dataset of pertubed images
    files_for_prediciton = [
        os.path.join("CelebA/perturb_protocol", f"image_names.txt")
    ]
    perturbed_images_direction = '/home/user/chuber/attribute-cam/result/myresult'
    dataset_perturb = attribute_cam.CelebA(files_for_prediciton,
                                           perturbed_images_direction,
                                           number_of_images=10)

    affact = attribute_cam.AFFACT(args.model_type,
                                  "cuda" if args.gpu else "cpu")

    affact.predict_all(dataset_perturb,
                       './result/myresult/Prediction-perturb.csv')





    with open("CelebA/perturb_protocol/image_names.txt", 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    print(image_paths)
    for img_name in image_paths[:args.image_count]:
        # img, x = load_img(img_path) # load original image
        # masks = generate_masks(N, s, p1)

        # # Save perturbed images
        img_name_no_ext, _ = os.path.splitext(img_name)
        # apply_and_save_mask(img, x, masks, "result/myresult", img_name_no_ext, N)

        # Generate saliency map
        class_idx = 0  # Adjust this to the desired target class index
        saliency_map = generate_saliency_map(img_name_no_ext, x, masks)

        # Save the saliency map
        plt.imshow(saliency_map, cmap='jet', alpha=0.7)
        plt.axis("off")
        plt.colorbar()
        print(img_name_no_ext)
        plt.savefig(f"result/myresult/saliency_map_{img_name_no_ext}.png", bbox_inches='tight')
        plt.close()

    print(f'The perturbation finished within: {datetime.now() - startTime}')


if __name__ == '__main__':
    main()
