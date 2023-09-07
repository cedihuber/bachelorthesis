import cv2 
import os
import argparse
from get_attributes import get_attr
import pandas as pd
import numpy as np
from tqdm import tqdm

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    csv = args.csv

    with open(csv, 'r') as f1: 
        df = pd.read_csv(f1)
        f1.close()
        
    df_pr_pos = df[df['prediction'] > 0]
    df_pr_neg = df[df['prediction'] < 0]

    df_gt_pos = df[df['ground truth'] == 1]
    df_gt_neg = df[df['ground truth'] == -1]

    df_true_pos = df[(df['prediction'] > 0) & (df['ground truth'] == 1)]
    df_false_pos = df[(df['prediction'] > 0) & (df['ground truth'] == -1)]
    df_true_neg = df[(df['prediction'] < 0) & (df['ground truth'] == -1)]
    df_false_neg = df[(df['prediction'] < 0) & (df['ground truth'] == 1)]

    df_correct = df[((df['prediction'] > 0) & (df['ground truth'] == 1)) | ((df['prediction'] < 0) & (df['ground truth'] == -1))]
    df_incorrect = df[((df['prediction'] > 0) & (df['ground truth'] == -1)) | ((df['prediction'] < 0) & (df['ground truth'] == 1))]

    attribute_names = get_attr()

    for a in tqdm(attribute_names):
        # prediction positive
        df_pr_pos_a = df_pr_pos[df_pr_pos['attribute name'] == a]
        column = df_pr_pos_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'pr_pos\avg_pr_pos_{a}.jpg'), avg_cam_on_img)

        # prediction negative
        df_pr_neg_a = df_pr_neg[df_pr_neg['attribute name'] == a]
        column = df_pr_neg_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'pr_neg\avg_pr_neg_{a}.jpg'), avg_cam_on_img)
        
        # ground truth 1
        df_gt_pos_a = df_gt_pos[df_gt_pos['attribute name'] == a]
        column = df_gt_pos_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'gt_pos\avg_gt_pos_{a}.jpg'), avg_cam_on_img)
        
        # ground truth -1
        df_gt_neg_a = df_gt_neg[df_gt_neg['attribute name'] == a]
        column = df_gt_neg_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'gt_neg\avg_gt_neg_{a}.jpg'), avg_cam_on_img)
        
        # true positive
        df_true_pos_a = df_true_pos[df_true_pos['attribute name'] == a]
        column = df_true_pos_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'true_pos\avg_true_pos_{a}.jpg'), avg_cam_on_img)
        
        # false positive
        df_false_pos_a = df_false_pos[df_false_pos['attribute name'] == a]
        column = df_false_pos_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'false_pos\avg_false_pos_{a}.jpg'), avg_cam_on_img)
        
        # true negative
        df_true_neg_a = df_true_neg[df_true_neg['attribute name'] == a]
        column = df_true_neg_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'true_neg\avg_true_neg_{a}.jpg'), avg_cam_on_img)
        
        # false negative
        df_false_neg_a = df_false_neg[df_false_neg['attribute name'] == a]
        column = df_false_neg_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'false_neg\avg_false_neg_{a}.jpg'), avg_cam_on_img)
        
        # sign(prediction) == sign(ground truth)
        df_correct_a= df_correct[df_correct['attribute name'] == a]
        column = df_correct_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'correct\avg_correct_{a}.jpg'), avg_cam_on_img)
        
        # sign(prediction) != sign(ground_truth)
        df_incorrect_a = df_incorrect[df_incorrect['attribute name'] == a]
        column = df_incorrect_a['image number']
        imgs = column.values.tolist()
        activations = []
        for im in tqdm(imgs, leave=False):
            activations.append(cv2.imread(os.path.join(input_dir, a, str(im), f'{im}_{a}_cam_on_img.jpg'), cv2.IMREAD_COLOR))
        avg_cam_on_img = np.round(np.mean(np.array(activations, dtype=np.uint8), axis=0))
        cv2.imwrite(os.path.join(output_dir, rf'incorrect\avg_incorrect_{a}.jpg'), avg_cam_on_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Get average of CAMs')
    parser.add_argument(
        'input_dir',
        type=str,
        help="Path to folder containing the GradCAMs for every image"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Path to folder where the output images should be stored"
    )
    parser.add_argument(
        'csv',
        type=str,
        help="Path to the csv file containing the data"
    )
    args = parser.parse_args
    main(args)
