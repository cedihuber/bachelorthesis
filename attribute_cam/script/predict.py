import csv
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import attribute_cam
#from get_shifted_landmarks import get_shifted_landmarks_df
 
 
def command_line_options():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Goes through the dataset and predicts the attributes")
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
      '-m', '--model-type',
      default='balanced',
      choices=['balanced', 'unbalanced'],
      help="Can be balanced or unbalanced"
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
  os.makedirs(args.output_directory, exist_ok=True)
  # create dataset
  file_lists = [os.path.join(args.protocol_directory, f"aligned_224x224_{args.which_set}_filtered_0.1.txt")]
  output_file = attribute_cam.prediction_file(args.output_directory, args.which_set, args.model_type)
  dataset = attribute_cam.CelebA(
      file_lists,
      args.source_directory,
      number_of_images=args.image_count
  )
 
 
  print(f"Predicting attributes for {len(dataset)} images with the {args.model_type} model")
 
  # create CAM module
  affact = attribute_cam.AFFACT(args.model_type, "cuda" if args.gpu else "cpu")
 
  startTime = datetime.now()
 
  affact.predict_all(dataset, output_file)
 
  print(f'The prediction finished within: {datetime.now() - startTime}')
  print(f'Wrote {output_file}')