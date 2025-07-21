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
      default=['balanced'],
      choices=['unbalanced', 'balanced'],
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
      default = ["pr=1", "pr=-1"],
      choices = list(attribute_cam.FILTERS.keys()),
      help="Average cams images with the given filters"
  )
  parser.add_argument(
      "-e", "--prop-energy",
      action="store_false",
      help="Disable proportional energy computation (use AMR instead)"
  )
  parser.add_argument(
      "-n", "--normalize-error",
      action="store_true",
      help="Compute normalized energy or AMR values"
  )
  parser.add_argument(
      "-l", "--latex-file",
      default="a_prop_result_table.tex",
      help="Select the file where to write errors into"
  )
  parser.add_argument(
    "-O", "--overlay-image",
    help="If provided, the masks will be overlayed to the given image and saved to the --mask-directory"
  )
  parser.add_argument(
    "-M", "--mask-directory",
    default="masks",
    help="Select the directory where to save the masks into"
  )
  args = parser.parse_args()

  return args


def main():
  args = command_line_options()

  # obtain list file containing the data
  file_lists = [os.path.join(args.protocol_directory, f"aligned_224x224_{args.which_set}_filtered_0.1.txt")]

  # read ground truth
  ground_truth_file = os.path.join(args.protocol_directory, "list_attr_celeba.txt")
  ground_truth = attribute_cam.read_list(ground_truth_file, " ", 2)


  # create masks
  masks, mask_sizes = attribute_cam.get_masks()
  #attribute_cam.write_masks(masks,"../../../../local/scratch/datasets/CelebA/aligned_224x224/183462.png" , args.mask_directory)
  if args.overlay_image is not None:
    # write masks overlayed with the given image
    attribute_cam.write_masks(masks, os.path.join(args.source_directory, args.overlay_image), args.mask_directory)

  # compute means and stds of AMR for various filters
  startTime = datetime.now()
  means, stds = {}, {}

  for model_type in args.model_types:
    # read predictions
    prediction_file = attribute_cam.prediction_file(args.output_directory, args.which_set, model_type)
    prediction = attribute_cam.read_list(prediction_file, ",", 0)
    cam_directory = "../../../../local/scratch/chuber/result/own_perturbation_4000masks_squared_0_85" #os.path.join(args.output_directory, model_type, args.cam_type)

    # create dataset
    dataset = attribute_cam.CelebA(
        file_lists,
        args.source_directory,
        cam_directory,
        args.image_count,
        args.attributes
    )

    for filter_type in args.filters:

      dataset.filter_type=filter_type

      # define filters and masks
      filter = attribute_cam.Filter(ground_truth, prediction, filter_type)

      print(f"Analyzing CAMS of type {args.cam_type} for {filter_type} filter, model {model_type} and {len(dataset.attributes)} attributes")

      # compute acceptable mask ratios
      print(args.prop_energy)
      stats = attribute_cam.statisics(dataset, filter, masks, mask_sizes, prop_energy=True)#args.prop_energy

      means[(model_type,filter_type)] = stats[0]
      stds[(model_type,filter_type)] = stats[1]

  print(f'The computation of statistics finished within: {datetime.now() - startTime}')

  print(f"Computing error rates")
  errors = {}
  for model_type in args.model_types:
    prediction_file = attribute_cam.prediction_file(args.output_directory, args.which_set, model_type)
    prediction = attribute_cam.read_list(prediction_file, ",", 0)
    errors[model_type] = attribute_cam.error_rate(dataset, ground_truth, prediction) # we exclude TER here


  # compute positive rate
  index_list = os.path.join(args.protocol_directory, "list_eval_partition.txt")
  indexes = attribute_cam.read_list(index_list, " ", 0, split_attributes=False)
  counts = attribute_cam.class_counts(args.attributes or attribute_cam.ATTRIBUTES, ground_truth, indexes)

  # sort attributes by counts
  sorted_attributes = [a[1] for a in sorted(((min(counts[a]), a) for a in dataset.attributes), reverse=True)]

  index = 1 if args.normalize_error else 0

  # write table
  table = [
      # Attributes
      [attribute.replace("_"," ")] +
      # Relative counts
      [counts[attribute][0] / sum(counts[attribute])] +
      # error rates on unbalanced model, including highlights
     # [f"\\bf {e:#.3f}" if i == 0 and counts[attribute][0] < counts[attribute][1] or i == 1 and counts[attribute][0] > counts[attribute][1] else f"{e:#.3f}" for i,e in enumerate(errors["unbalanced"][attribute][:2]) if "unbalanced" in args.model_types] +
      # error rates on balanced model
      [e for e in errors["balanced"][attribute][:2] if "balanced" in args.model_types] +
      # proportional energy on unbalanced model, including highlights
      # [f"\\bf {means[('unbalanced',filter_type)][attribute][index]:#.3f}" if filter_type == "pr=-1" and counts[attribute][0] < counts[attribute][1] or filter_type == "pr=1" and counts[attribute][0] > counts[attribute][1] else f"{means[('unbalanced',filter_type)][attribute][index]:#.3f}" for filter_type in args.filters if "unbalanced" in args.model_types] +
      # proportional energy on balanced model
      [means[("balanced",filter_type)][attribute][index] for filter_type in args.filters if "balanced" in args.model_types]
      # for all attributes
      for attribute in sorted_attributes
  ]

  print(tabulate.tabulate(table, headers = ["Attribute","Positives"] + ["FNR", "FPR"] * len(args.model_types) + args.filters * len(args.model_types), floatfmt="#.3f"))

  with open(f"{cam_directory}/{args.latex_file}", "w") as w:
    w.write(tabulate.tabulate(table,tablefmt="latex_raw", floatfmt="#.3f"))

if(__name__ == "__main__"):
    main()