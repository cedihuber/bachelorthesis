import argparse
import os
from datetime import datetime
import attribute_cam
import tabulate


def command_line_options():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Extracts CAM images for all images from the given dataset using the given cam technique and the given model")
  parser.add_argument(
      '-w', '--which-set',
      default = "validation",
      choices = ("validation", "test"),
      help="Select to process the given part(s) of the dataset"
  )
  parser.add_argument(
      '-s', '--source-directory',
      default='/local/scratch/datasets/CelebA/aligned_224x224',
      help="Select directory containing the input dataset"
  )
  parser.add_argument(
      '-p', '--protocol-directory',
      default='/local/scratch/datasets/CelebA/protocol',
      help="Select directory containing the original filelists defining the protocol and ground truth of CelebA"
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
      '-f',
      '--filters',
      nargs="*",
      default = list(attribute_cam.FILTERS.keys()),
      choices = list(attribute_cam.FILTERS.keys()),
      help="Average cams images with the given filters"
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

  # obtain list file containing the data
  file_lists = [f"files/aligned_224x224_{args.which_set}_filtered_0.1.txt"]
  cam_directory = os.path.join(args.output_directory, args.model_type, args.cam_type)

  # read ground truth and predictions
  ground_truth_file = os.path.join(args.protocol_directory, "list_attr_celeba.txt")
  ground_truth = attribute_cam.read_list(ground_truth_file, " ", 2)
  prediction_file = attribute_cam.prediction_file(args.output_directory, args.which_set, args.model_type)
  prediction = attribute_cam.read_list(prediction_file, ",", 0)


  # create masks
  masks, mask_sizes = attribute_cam.get_masks()

  # create dataset
  dataset = attribute_cam.CelebA(
      file_lists,
      args.source_directory,
      cam_directory,
      args.image_count,
      args.attributes
  )

  # compute means and stds of AMR for various filters
  startTime = datetime.now()
  amr_means, amr_stds = {}, {}
  for filter_type in args.filters:

    dataset.filter_type=filter_type

    # define filters and masks
    filter = attribute_cam.Filter(ground_truth, prediction, filter_type)

    print(f"Analyzing CAMS of type {args.cam_type} for {filter_type} filter and {len(dataset.attributes)} attributes")

    # compute acceptable mask ratios
    means, stds = attribute_cam.amr_statisics(dataset, filter, masks, mask_sizes)

    amr_means[filter_type] = means
    amr_stds[filter_type] = stds

  print(f'The computation of statistics finished within: {datetime.now() - startTime}')

  # compute positive rate
  index_list = os.path.join(args.protocol_directory, "list_eval_partition.txt")
  indexes = attribute_cam.read_list(index_list, " ", 0, split_attributes=False)
  counts = attribute_cam.class_counts(args.attributes or attribute_cam.ATTRIBUTES, ground_truth, indexes)

  # compute error rate
  print(f"Computing error rates")
  errors = attribute_cam.error_rate(dataset, ground_truth, prediction)


  # write table
  table = [
      [attribute] +
      [f"{counts[attribute][0] / sum(counts[attribute]):1.3f}"] +
      [f"{e:1.3f}" for e in errors[attribute]] +
      [
          f"{amr_means[filter_type][attribute][0]:1.3f}" for filter_type in args.filters
      ] for attribute in dataset.attributes
  ]

  print(tabulate.tabulate(table, headers = ["Attribute", "Positives", "FNR", "FPR", "Error"] + args.filters))
