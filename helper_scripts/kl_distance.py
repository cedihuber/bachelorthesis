from matplotlib.image import imread
import argparse
from get_attributes import get_attr
import scipy.special as ss
import os
import csv
import numpy as np

def kl_div(balanced, unbalanced):
    res = sum(ss.kl_div(balanced, unbalanced))
    return res
    

def main(args): 
    attributes = get_attr()
    
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['attribute', 'KL(balanced||unbalanced)'])

        for a in attributes:
            # image to matrix
            balanced = np.ndarray.flatten(imread(os.path.join(args.in_b, f'avg_gray_new_{a}.jpg')))
            unbalanced = np.ndarray.flatten(imread(os.path.join(args.in_ub, f'avg_gray_new_{a}.jpg')))
            if np.max(balanced) > 1:
                balanced = balanced/np.max(balanced)
            if np.max(unbalanced) > 1:
                unbalanced = unbalanced/np.max(unbalanced)
            # calculate kl
            kl_div_b_ub = kl_div(balanced, unbalanced)
            writer.writerow([a,
                             kl_div_b_ub])
        f.close()
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate KL distance.")
    parser.add_argument(
        'in_b',
        type=str,
        help="Path to average GradCAMs from balanced network."
    )
    parser.add_argument(
        'in_ub',
        type=str,
        help="Path to average GradCAMs from unbalanced network."
    )
    parser.add_argument(
        'out',
        type=str,
        help="Path to file where KL distance should be stored in."
    )
    args = parser.parse_args()
    main(args)