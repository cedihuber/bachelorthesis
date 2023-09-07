import argparse
import pandas as pd
from get_attributes import get_attr
import csv

def main(args):
    with open(args.input, 'r') as f1: 
        df = pd.read_csv(f1)
        f1.close()
    
    attribute_names = get_attr()
    tot = 0
    with open(args.output, 'w', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(['attribute name', 'amr corrected'])
        for a in range(40):
            # select rows of current attribute
            df_attr = df[df['attribute name'] == attribute_names[a]]
            # select rows where ground truth == 1
            df_gt_true = df_attr[df_attr['ground truth'] == 1]
            # calculate average of amr
            avg = df_gt_true['amr corrected'].mean()
            writer.writerow([attribute_names[a],
                             avg])
            tot += avg
        overall = tot/40
        writer.writerow(['OVERALL',
                             overall])
        f2.close()   
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get average acceptable mask ratio for the case of presence of an attribute")
    parser.add_argument(
        'input',
        type=str,
        help="Path to file containing the amr values for each image"
    )
    parser.add_argument(
        'output',
        type=str,
        help="Path to output file"
    )
    args = parser.parse_args()
    main(args)