import argparse
import os
from datetime import datetime
import attribute_cam
import tabulate
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages



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
      default=['balanced', 'unbalanced'],
      choices=['balanced', 'unbalanced'],
      help="Can be balanced or unbalanced"
  )
  parser.add_argument(
      "-l", "--latex-file",
      default="error_rates.tex",
      help="Select the file where to write errors into"
  )
  parser.add_argument(
      '-P', "--pdf-file",
      default="error_rates.pdf",
      help = "Select the file to include the plots"
  )
  args = parser.parse_args()

  return args


def main():
  args = command_line_options()

  # obtain list file containing the data
  file_lists = [os.path.join(args.protocol_directory, f"files/aligned_224x224_{args.which_set}_filtered_0.1.txt")]
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
      [attribute.replace("_"," ")] +
      [counts[attribute][0] / sum(counts[attribute])] +
      [f"\\bf {e:#.3f}" if i == 0 and counts[attribute][0] < counts[attribute][1] or i == 1 and counts[attribute][0] > counts[attribute][1] else f"{e:#.3f}" for i,e in enumerate(errors["unbalanced"][attribute]) if "unbalanced" in args.model_types] +
      [e for e in errors["balanced"][attribute] if "balanced" in args.model_types]
      for attribute in sorted_attributes
  ]

  print(tabulate.tabulate(table, headers = ["Attribute", "Positives"] + ["FNR", "FPR", "Error"]*len(args.model_types), floatfmt="#.3f"))

  with open(args.latex_file, "w") as w:
    w.write(tabulate.tabulate(table,tablefmt="latex_raw", floatfmt="#.3f"))


  # create plot
  print(f"Plotting")
  pdf = PdfPages(args.pdf_file)

  try:
    colors = ["red", "green"]
    labels = ["False Negative Rate", "False Positive Rate"]
    for model_type in args.model_types:
      fig = pyplot.figure(figsize=(20,5))
      for i, attribute in enumerate(sorted_attributes):
        majority = 0 if counts[attribute][0] > counts[attribute][1] else 1
        pyplot.bar([i], [errors[model_type][attribute][1-majority]], color=colors[majority], align="center", label=labels[majority] if not i else None)
        pyplot.bar([i], [-errors[model_type][attribute][majority]], color=colors[1-majority], align="center", label=labels[1-majority] if not i else None)

      pyplot.xticks(range(len(sorted_attributes)), [" " + a.replace("_", " ") for a in sorted_attributes], va="top", ha='center', rotation=90., size=14)
      pyplot.xlim([-.5,len(sorted_attributes)-.5])
#      pyplot.ylabel("Prediction Error", size=12)
      pyplot.text(-2,.15, "MinPE", rotation=90, va="center", size=14)
      pyplot.text(-2,-.15, "MajPE", rotation=90, va="center", size=14)
      pyplot.legend(loc="upper right", ncols=2, prop={"size":14})
      pyplot.tight_layout()

      pdf.savefig(fig, pad_inches=0)
  finally:
    pdf.close()
