import argparse
import os

def main(args):
    # enlist all images in folder
    imgs = os.listdir(args.input_dir)
    img_numbers = [i.split('.')[0] for i in imgs]
      
    # write image numbers in a new txt file
    with open(args.output_dir, 'w', newline='') as f:
        for n in img_numbers:
            f.write(str(n) + '.png')
            f.write('\n')
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a textfile containing all the image numbers.")
    parser.add_argument(
        'input_dir',
        type=str,
        help="Path to folder containing the frontal pose images of the CelebA dataset"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Path to the textfile where the output should be stored"
    )
    args = parser.parse_args()
    main(args)