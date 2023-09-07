import pandas as pd

def get_landmarks_df(start, input_file):
    with open(input_file, 'r', newline='') as f:
        # get data
        n = ['image', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
        df_aligned = pd.read_csv(f, sep=' ', names=n, skipinitialspace=True, skiprows=1)
        f.close()
        df_aligned = df_aligned[(start-1):202599]
        
        return df_aligned
        
def get_landmarks_img(df, img_number):
    # get landmarks for current image
    landmarks_aligned = df[df['image'] == f'{img_number}.jpg']
    # convert to list
    landmarks_aligned = landmarks_aligned.values.tolist().pop()
    # remove {img_number}.jpg
    landmarks_aligned = landmarks_aligned[1:]
    
    return landmarks_aligned