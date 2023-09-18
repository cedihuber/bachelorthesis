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
      '-m', '--model-types',
      nargs="+",
      default=['balanced', 'unbalanced'],
      choices=['balanced', 'unbalanced'],
      help="Can be balanced or unbalanced"
  )
  parser.add_argument(
      "-l", "--latex-file",
      default="error_rates.tex",
      help="Select the file where to write errors into"
  )
  args = parser.parse_args()

  return args


def main():
  args = command_line_options()

  # obtain list file containing the data
  file_lists = [f"files/aligned_224x224_{args.which_set}_filtered_0.1.txt"]
#  cam_directory = os.path.join(args.output_directory, args.model_type, args.cam_type)

  # read ground truth and predictions
  ground_truth_file = os.path.join(args.protocol_directory, "list_attr_celeba.txt")
  ground_truth = attribute_cam.read_list(ground_truth_file, " ", 2)

  # create dataset
  dataset = attribute_cam.CelebA(
      file_lists,
      args.source_directory,
      None,
      args.image_count,
      args.attributes
  )

  # compute positive rate
  index_list = os.path.join(args.protocol_directory, "list_eval_partition.txt")
  indexes = attribute_cam.read_list(index_list, " ", 0, split_attributes=False)
  counts = attribute_cam.class_counts(args.attributes or attribute_cam.ATTRIBUTES, ground_truth, indexes)

  # sort attributes by counts
  sorted_attributes = [a[1] for a in sorted(((min(counts[a]), a) for a in dataset.attributes), reverse=True)]

  # compute error rate
  print(f"Computing error rates")
  errors = {}
  for model_type in args.model_types:
    prediction_file = attribute_cam.prediction_file(args.output_directory, args.which_set, model_type)
    prediction = attribute_cam.read_list(prediction_file, ",", 0)
    errors[model_type] = attribute_cam.error_rate(dataset, ground_truth, prediction)

  # write table
  table = [
      [attribute] +
      [f"{counts[attribute][0] / sum(counts[attribute]):1.3f}"] +
      [f"{e:1.3f}" for model_type in args.model_types for e in errors[model_type][attribute]]
      for attribute in sorted_attributes
  ]

  print(tabulate.tabulate(table, headers = ["Attribute", "Positives"] + ["FNR", "FPR", "Error"]*len(args.model_types)))

  with open(args.latex_file, "w") as w:
    w.write(tabulate.tabulate(table,tablefmt="latex_raw"))
