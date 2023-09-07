import argparse
import os
from get_landmarks import get_landmarks_df
from tqdm import tqdm

def main(args):
    # get landmarks as pandas dataframe
    df = get_landmarks_df(args.start, args.path_landmarks_in)
    input_dir = args.input_dir
    c = args.start
      
    # write the shifted landmarks in a new txt file
    with open(args.output_dir, 'w', newline='') as f:
        header = 'image lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y'
        f.write(header)
        f.write('\n')
        for i in tqdm(range(args.nr_imgs)):
            path = os.path.join(input_dir, f'{c}.png')
            c += 1
            img_number = path.split("\\")[-1].split('.')[0]
            
            # get landmarks for current image
            landmarks_aligned = df[df['image'] == f'{img_number}.jpg']
            # convert to list
            landmarks_aligned = landmarks_aligned.values.tolist().pop()
            # remove {img_number}.jpg
            landmarks_aligned = landmarks_aligned[1:]
            
            # enlist shifted values
            landmarks_shifted = ''
            landmarks_shifted += str(img_number) + '.jpg '
            for j in range(10):
                # x values
                if (j % 2):
                    landmarks_shifted += str(landmarks_aligned[j] - 10)
                # y values
                else:
                    landmarks_shifted += str(landmarks_aligned[j] + 24)
                landmarks_shifted += ' '
            
            f.write(landmarks_shifted)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="draw points")
    parser.add_argument(
        'input_dir',
        type=int,
        help="Path to folder containing the images of the CelebA dataset (aligned_224x224)"
    )
    parser.add_argument(
        'output_dir',
        type=int,
        help="Path to the textfile where the shifted landmarks should be stored"
    )
    parser.add_argument(
        'path_landmarks_in',
        type=int,
        help="Path to the textfile containing the original landmarks (list_landmarks_align_celeba.txt)"
    )
    parser.add_argument(
        'start',
        type=int,
        help="Number of first image"
    )
    parser.add_argument(
        'nr_imgs',
        type=int,
        help="Number of input images"
    )
    args = parser.parse_args()
    main(args)