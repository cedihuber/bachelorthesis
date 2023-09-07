import argparse
import pandas as pd
from get_attributes import get_attr
import csv

def main(args):
    with open(args.input, 'r') as f1: 
        df = pd.read_csv(f1)
        f1.close()
    
    attributes = get_attr()
    tot = 0
    with open(args.output, 'w', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(['attribute name', 'amr corrected'])
        for a in range(40):
            sub = df[df['attribute name'] == attributes[a]]
            avg = sub['amr corrected'].mean()
            writer.writerow([attributes[a],
                             avg])
            tot += avg
        overall = tot/40
        writer.writerow(['OVERALL',
                             overall])
        
        f2.close()   
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get averaged acceptable mask ratio for each attribute respectively")
    parser.add_argument(
        'input',
        type=str,
        help="Path to alanysis file"
    )
    parser.add_argument(
        'output',
        type=str,
        help="Path to output file"
    )
    args = parser.parse_args()
    main(args)