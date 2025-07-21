import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
import torch
import torchvision
import numpy as np
import tabulate
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
      '-s1', '--source-directory-reference',
      default='../../../../local/scratch/chuber/result/own_perturbation_4000masks_squared_0_75',
      help="Select directory containing the input dataset"
  )
  
  parser.add_argument(
      '-s2', '--source-directory-comparison',
      default='../../../../local/scratch/chuber/result/own_perturbation_4000masks_squared_0_85',
      help="Select directory containing the input dataset"
  )
  parser.add_argument(
      '-p', '--protocol-directory',
      default='CelebA/protocol',
      help="Select directory containing the original filelists defining the protocol and ground truth of CelebA"
  )
  parser.add_argument(
      '-o', '--output-directory',
      default="../../../../local/scratch/chuber/result/rise_testing_new",
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
  parser.add_argument(
      "-l", "--latex-file",
      default="../../../../local/scratch/chuber/result/own_perturbation_4000masks_squared_0_75/cosine_similarity_balanced_unbalanced.tex",
      help="Select the file where to write errors into"
  )
  args = parser.parse_args()

  return args


def load_average(name):
    # overlay = torchvision.io.image.read_image(name).numpy().transpose(1,2,0)
    activation = np.load(name + ".npy")
    return torch.tensor(activation, dtype=torch.float32).flatten()

def is_invalid_vector(vec):
    return not torch.isfinite(vec).all() or torch.norm(vec) == 0

def main():
  args = command_line_options()
  results = {attr: {} for attr in attribute_cam.ATTRIBUTES}
  for filter_type in args.filters:
  # create dataset
    for attribute in attribute_cam.ATTRIBUTES:
      reference = load_average(os.path.join(args.source_directory_reference, args.which_set+"-"+filter_type, attribute+".png"))
      comparison = load_average(os.path.join(args.source_directory_comparison, args.which_set+"-"+filter_type, attribute+".png"))
      if is_invalid_vector(reference) or is_invalid_vector(comparison):
        print(f"Warning: Invalid (NaN, inf, or zero) vector for {attribute} with filter {filter_type}")
      cosine_similarity = torch.nn.functional.cosine_similarity(reference.unsqueeze(0), comparison.unsqueeze(0))
      
      #print("Cosine similarity:", cosine_similarity.item())
      results[attribute][filter_type] = cosine_similarity.item()
      
  headers = ["Attribute"] + args.filters
  table = [
      [attribute.replace("_", " ")] + [f"{results[attribute].get(ft, 'N/A'):.3f}" for ft in args.filters]
      for attribute in sorted(attribute_cam.ATTRIBUTES)
  ]  

  print(tabulate.tabulate(table, headers=headers, floatfmt=".3f")) 
  os.makedirs(os.path.dirname(args.latex_file), exist_ok=True)
  # Write to LaTeX
  with open(args.latex_file, "w") as f:
      f.write(tabulate.tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".3f"))

if __name__ == "__main__":
    main()
