import pandas as pd

def get_shifted_landmarks_df(path): # path to textfile containing the shifted landmarks (landmarks_aligned_shifted.txt)
    with open(path, 'r', newline='') as f:
        # get data
        n = ['image', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
        df_shifted = pd.read_csv(f, sep=' ', names=n, skipinitialspace=True, skiprows=1, index_col=False)
        f.close()
        
        return df_shifted
        
def get_shifted_landmarks_img(df, img_number):
    # get landmarks for current image
    landmarks_shifted = df[df['image'] == f'{img_number}.jpg']
    # convert to list
    landmarks_shifted = landmarks_shifted.values.tolist().pop()
    # remove {img_number}.jpg
    landmarks_shifted = landmarks_shifted[1:]
    
    return landmarks_shifted
