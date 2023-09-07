from get_attributes import get_attr
import pandas as pd 

def get_ground_truth(path):
    with open(path, 'r', newline='') as f:
        # get ground truth line by line as a pandas dataframe
        a = get_attr()
        a.insert(0, 'image')
        df = pd.read_csv(f, sep=' ', skipinitialspace=True, header=None, skiprows=1)
        f.close()
        df.columns = a
            
    return df