import csv
import argparse
import os
from grad_cam import gen_gradcam
from predict import load_model, preprocess_img, predict
from get_attributes import get_attr
from ground_truth import get_ground_truth
from error_rate import calc_error_rate
from datetime import datetime
from tqdm import tqdm
from acceptable_mask_ratio import get_accetptable_mask_ratio
from mask_32 import get_masks
import pandas as pd
#from get_shifted_landmarks import get_shifted_landmarks_df

def oppositeSigns(x, y):
    return ((x*y) < 0)

def is_false_pos(gt, c): # ground truth -1, classification > 0
    return (gt<c)

def is_false_neg(gt, c): # ground truth 1, classification < 0
    return (gt>c)

def main(args):
    startTime = datetime.now()
    
    # check if output folder exists, if not, create it
    output_dir = rf'D:\{args.run}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # make list with image paths
    input_dir = args.input_dir
    with open(input_dir, 'r') as f1:
        imgs_filtered = f1.readlines()
        imgs_filtered = [(nr.split('.')[0]+'.png') for nr in imgs_filtered]
        f1.close()
       
    dir_imgs = 'C:\\Users\\johannab\\ba_johanna\\aligned_224x224'
    imgs_all = os.listdir(dir_imgs)
    #print(imgs_all[:10])
    img_numbers = [nr for nr in imgs_all if nr in imgs_filtered]
    folder_imgs = input_dir.split('\\')[-1].split('.txt')[0]
    img_paths = [os.path.join(dir_imgs, nr) for nr in img_numbers]
    img_numbers = [int(nr.split('.')[0]) for nr in img_numbers]
    
    start = img_numbers[0]
    end = img_numbers[-1]
    
    
    ### PART 1: generate CAMs
    startTime2 = datetime.now()
    
    # dictionary with attribute names
    attribute_names = get_attr()

    for a in tqdm(range(40)): # iterate over all 40 attributes
        output_dir_gradcam = os.path.join(output_dir, args.model_type, args.layer, 'gradcam', attribute_names[a])
        # generate CAMs for all the images for the current attribute
        gen_gradcam(img_paths, img_numbers, args.model_type, a, start, end, args.nr_imgs, output_dir_gradcam, args.use_cuda, args.path_gt)
        
    print(f'The generation of CAMs finished within: {datetime.now() - startTime2}')
    
    
    ### PART 2: generate csv file with data     
    # get masks and respective sizes
    dict_masks, dict_mask_sizes = get_masks()
    # load module and device
    network, device = load_model(args.model_type)
    path_csv = os.path.join(output_dir, f'list_img_analysis_{folder_imgs}.csv')
    with open(path_csv, 'w', newline='') as f3: 
        writer = csv.writer(f3)
        writer.writerow(['image number', 'attribute number', 'attribute name', 'prediction', 'ground truth', 'amr', 'amr corrected', 'relative mask size', 'activated pixels', 'pixels within mask', 'pixels outside mask','error', 'prediction false pos', 'prediction false neg'])
        
        # for each image: classify all 40 attributes and save data in csv file
        ground_truth = get_ground_truth(args.path_gt)
        #ground_truth = ground_truth[(162771-1):202599]
        for path in tqdm(img_paths): # iterate over all the images
            
            # extract img number
            img_number = path.split("\\")[-1].split('.')[0]
                
            # classify all attributes for current image 
            tensor = preprocess_img(path, device)   
            prediction_values = predict(network, tensor)
            
            # get ground truth values for current image
            ground_truth_values = ground_truth[ground_truth['image'] == f'{img_number}.jpg']
            # convert to list
            ground_truth_values = ground_truth_values.values.tolist().pop()
            # remove {img_number}.jpg
            ground_truth_values = ground_truth_values[1:]
            
            # iterate over attributes
            for i in range(40):                
                # calculate acceptable mask ratio
                output_dir_mask = os.path.join(output_dir, args.model_type, args.layer, 'gradcam', attribute_names[i], img_number)
                amr, amr_corr, mask_relative, activated_pxls, activated_in, activated_out = get_accetptable_mask_ratio(path, img_number, output_dir_mask, attribute_names[i], dict_masks[i], dict_mask_sizes[i])
                # determine whether the current attribute has been classified correctly for the current image
                error = 0
                false_pos = 0
                false_neg = 0
                if oppositeSigns(prediction_values[i], float(ground_truth_values[i])):
                    error = 1
                    if is_false_pos(float(ground_truth_values[i]), prediction_values[i]):
                        false_pos = 1
                    if is_false_neg(float(ground_truth_values[i]), prediction_values[i]):
                        false_neg = 1
                # write csv file
                writer.writerow([img_number, 
                                 i, 
                                 attribute_names[i], 
                                 prediction_values[i], 
                                 ground_truth_values[i],
                                 amr,
                                 amr_corr,
                                 mask_relative,
                                 activated_pxls,
                                 activated_in,
                                 activated_out,
                                 error,
                                 false_pos,
                                 false_neg])
        f3.close()

        ### PART 3: error rate       
        # calculate error rate
        calc_error_rate(path_csv, os.path.join(output_dir, f'error_rate_{folder_imgs}.csv'))
        
    print(f'The whole analysis finished within: {datetime.now() - startTime}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with data for all given images, generate CAMs for all attributes for every image, calculate the error rate for every attribute")
    parser.add_argument(
        'input_dir',
        type=str,
        help="Path to folder containing the input images"
    )
    parser.add_argument(
        'path_gt',
        type=str,
        help="Path to the textfile containing the ground truth values"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Path to folder where the output should be stored"
    )
    parser.add_argument(
        'nr_imgs',
        type=int,
        help="Number of input images"
    )
    parser.add_argument(
        '--path_mask',
        type=int,
        help="Path to file containing the mask sizes"
    )
    parser.add_argument(
        'run',
        type=str,
        help="Name of run"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='balanced',
        choices=['balanced', 'unbalanced'],
        help="Can be balanced or unbalanced"
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='res5c',
        help="Layer of network to generate GradCAMs from"
    )
    parser.add_argument(
        '--use-cuda', 
        default=True,
        help='Use NVIDIA GPU acceleration'
    )
    args = parser.parse_args()
    main(args)