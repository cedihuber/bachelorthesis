import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
#from get_shifted_landmarks import get_shifted_landmarks_df


def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extracts CAM images for all images from the given dataset using the given cam technique and the given model")
    parser.add_argument(
        '-w', '--which-sets',
        default = ["validation"],
        nargs="+",
        choices = ("validation", "test"),
        help="Select to process the given part(s) of the dataset"
    )
    parser.add_argument(
        '-s', '--source-directory',
        default='/local/scratch/datasets/CelebA/aligned_224x224',
        help="Select directory containing the input dataset"
    )
    parser.add_argument(
        '-o', '--output-directory',
        default="./result",
        help="Path to folder where the output should be stored"
    )
    parser.add_argument(
        '-i', '--image-count',
        type=int,
        help="if given, limit the number of images"
    )
    parser.add_argument(
        '-a', '--attributes',
        nargs='+',
        choices=attribute_cam.ATTRIBUTES,
        help="Extract CAMS only for the given attributes"
    )
    parser.add_argument(
        '-m', '--model-type',
        default='balanced',
        choices=['balanced', 'unbalanced'],
        help="Can be balanced or unbalanced"
    )
    parser.add_argument(
        '-c', '--cam-type',
        default='grad-cam',
        choices=list(attribute_cam.SUPPORTED_CAM_TYPES.keys()),
        help="Select the type of CAM method that you want to apply"
    )
    parser.add_argument(
        '--gpu',
        action="store_false",
        help='Do not use GPU acceleration (will be **disabled** when selected)'
    )
    args = parser.parse_args()

    return args


def main():
    args = command_line_options()

    # create dataset
    file_lists = [f"files/aligned_224x224_{which}_filtered_0.1.txt" for which in args.which_sets]
    cam_directory = os.path.join(args.output_directory, args.model_type, args.cam_type)
    dataset = attribute_cam.CelebA(
        file_lists,
        args.source_directory,
        cam_directory,
        args.image_count,
        args.attributes
    )


    print(f"Generating CAMS of type {args.cam_type} for {len(dataset)} images and {len(dataset.attributes)} attributes")

    # create CAM module
    cam = attribute_cam.CAM(args.cam_type)
    affact = attribute_cam.AFFACT(args.model_type, "cuda" if args.gpu else "cpu")

    ### generate CAMs
    startTime = datetime.now()

    cam.generate_cam(affact,dataset,args.gpu)

    print(f'The generation of CAMs finished within: {datetime.now() - startTime}')
