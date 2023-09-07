import pandas as pd
import numpy as np
import argparse

def main(args):
    # get data
    df_error = pd.read_csv(args.error)
    list_error = df_error['error rate in %'].to_numpy()
    list_error = list_error*0.01
    # exclude the overall value
    list_error = list_error[:40]
    
    df_amr = pd.read_csv(args.amr)
    list_amr = df_amr['amr corrected'].to_numpy()
    # exclude the overall value
    list_amr = list_amr[:40]
    
    # compute the covariance matrix
    covmat_b = np.cov(list_amr, list_error, rowvar=False)
    print(covmat_b)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the covariance matrix for values of the average acceptable mask ratio and the error rate of either the balanced or the unbalanced network.")
    parser.add_argument(
        'error',
        type=str,
        help="Path to file containing the error rate"
    )
    parser.add_argument(
        'amr',
        type=str,
        help="Path to file containing the amr"
    )
    args = parser.parse_args()
    main(args)
    

