import argparse
import os
from datetime import datetime
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
      "-f", "--force",
      action="store_true",
      help="If selected, files will be overwritten if they already exist, otherwise existing files will be skipped"
    )
  parser.add_argument(
      '-pl', '--parallel',
      type=int,
      help="If selected, the extraction will run in the given number of parallel processes (on one GPU only)"
  )
  parser.add_argument(
      '-X', '--standard-target',
      action="store_true",
      help="Use the standard ClassifierTargetOutput for generating CAM images"
  )
  parser.add_argument(
      '--gpu',
    #   action="store_false",
      type=str,
      default="cuda:7",
      help='Do not use GPU acceleration (will be **disabled** when selected)'
  )
  args = parser.parse_args()
 
  return args
 
def _run_extraction(params):
  args, index = params
  global datasets
  dataset = datasets[index]
  print(f"Generating CAMS of type {args.cam_type} for {len(dataset)} images and {len(dataset.attributes)} attributes")
  # create CAM module
  if args.standard_target:
    import pytorch_grad_cam
    cam = attribute_cam.CAM(args.cam_type, pytorch_grad_cam.utils.model_targets.ClassifierOutputTarget)
  else:
    cam = attribute_cam.CAM(args.cam_type)
  # load AFFACT model
  affact = attribute_cam.AFFACT(args.model_type, args.gpu)
 
  # generate CAMs
  cam.generate_cam(affact,dataset,args.gpu,args.force)
 
 
 
def main():
  args = command_line_options()
 
  # create dataset
  file_lists = [os.path.join(args.protocol_directory, f"aligned_224x224_{which}_filtered_0.1.txt") for which in args.which_sets]
  cam_directory = os.path.join(args.output_directory, args.model_type, args.cam_type)
 
  startTime = datetime.now()
 
  global datasets
  if args.parallel is None:
    datasets = [attribute_cam.CelebA(
        file_lists,
        args.source_directory,
        cam_directory,
        args.image_count,
        args.attributes
    )]
    _run_extraction((args,0))
 
  else:
    datasets = attribute_cam.split_dataset(
        args.parallel,
        file_lists,
        args.source_directory,
        cam_directory,
        args.image_count,
        args.attributes
    )
 
    import multiprocessing
    pool = multiprocessing.Pool(args.parallel)
    pool.map(_run_extraction, [(args,i) for i in range(args.parallel)])
 
  print(f'The generation of CAMs finished within: {datetime.now() - startTime}')
 