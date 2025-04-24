import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
#from get_shifted_landmarks import get_shifted_landmarks_df


def command_line_options():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Averages CAM images extracted via `extract_cams.py` from the given dataset using the given cam technique and the given model")
  parser.add_argument(
      '-w', '--which-set',
      default = "test",
      choices = ("validation", "test"),
      help="Select to process the given part(s) of the dataset"
  )
  parser.add_argument(
      '-s', '--source-directory',
      default='CelebA/aligned_224x224',
      help="Select directory containing the input dataset"
  )
  parser.add_argument(
      '-p', '--protocol-directory',
      default='CelebA/protocol',
      help="Select directory containing the original filelists defining the protocol and ground truth of CelebA"
  )
  parser.add_argument(
      '-o', '--output-directory',
      default="../../../../local/scratch/chuber/result/rise_testing_save_new",
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
      '-f',
      '--filters',
      nargs="+",
      default = ["pr=1","pr=-1"],
      choices = list(attribute_cam.FILTERS.keys()),
      help="Average cams images with the given filters"
  )
  args = parser.parse_args()

  return args


def main():
  args = command_line_options()

  # get some directories
  file_lists = [os.path.join("../../../../local/scratch/chuber/result/rise_testing_save_new/img_names.txt")]
  cam_directory = args.output_directory
  # read ground truth and predictions
  ground_truth_file = os.path.join(args.protocol_directory, "list_attr_celeba.txt")
  ground_truth = attribute_cam.read_list(ground_truth_file, " ", 2)
  prediction_file = attribute_cam.prediction_file("./result", args.which_set, args.model_type)
  prediction = attribute_cam.read_list(prediction_file, ",", 0)

  startTime = datetime.now()

  # compute averages using several filters
  for filter_type in args.filters:

    # create dataset
    dataset = attribute_cam.CelebA(
        file_lists,
        args.source_directory,
        cam_directory,
        args.image_count,
        args.attributes,
        args.which_set + "-" + filter_type
    )

    # create filter based on the ground truth and predictions
    filter = attribute_cam.Filter(ground_truth, prediction, filter_type)

    print(f"Averaging RISE of type {args.cam_type} for {filter_type} filter and {len(dataset.attributes)} attributes")

    # compute average images
    attribute_cam.average_cam(dataset, filter)

  print(f'The averaging of CAMs finished within: {datetime.now() - startTime}')

if __name__ == "__main__":
    main()
