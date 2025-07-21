import pytorch_grad_cam #check this package to add methods
import numpy
import tqdm
import os
import torch


# The different functions to create CAM images
# TODO: Add more
SUPPORTED_CAM_TYPES={
  # add method elementwise cam 
  "grad-cam": pytorch_grad_cam.GradCAM,
  "score-cam": pytorch_grad_cam.ScoreCAM,
  "hirescam": pytorch_grad_cam.HiResCAM,
  "fullgrad_cam": pytorch_grad_cam.FullGrad,
  "grad_cam_plus_plus": pytorch_grad_cam.GradCAMPlusPlus,
  "element_wise_cam": pytorch_grad_cam.GradCAMElementWise
  
}


# Taken directly from the thesis of Bieri
class BinaryCategoricalClassifierOutputTarget:
  def __init__(self, category):
    self.category = category

  def __call__(self, model_output):
    if len(model_output.shape) == 1:
      return abs(model_output[self.category])
    return abs(model_output[:, self.category])

# computes the CAM images and saves them in the target directory
class CAM:
  def __init__(self, cam_type, output_target_class = BinaryCategoricalClassifierOutputTarget):
    # store the type of CAM algorithm that we want to use
    self.cam_class = SUPPORTED_CAM_TYPES[cam_type]
    self.output_target_class = output_target_class
 
 
  def generate_cam(self, affact_model, celeba_dataset, use_cuda=True, force=False):
    # instantiate the CAM algorithm for the given model
    # breakpoint()
    with self.cam_class(
      model = affact_model.model(),
      target_layers = affact_model.cam_target_layers(),
      # uses_gradients = use_cuda
    ) as cam:
 
      for image_index in tqdm.tqdm(celeba_dataset):
        # load image
        tensor, image = celeba_dataset.source_image(image_index)
        
 
        for attribute, index in celeba_dataset.attributes.items():
 
          if not os.path.isfile(celeba_dataset.cam_filename(attribute, image_index)) or force:
 
            # extract cam for the current attribute
            targets = [self.output_target_class(index)]
            activation = cam(tensor, targets)[0]
 
            # NOTE: The source image for this function is float in range [0,1]
            # the ouput of it is uint8 in range [0,255]
            overlay = pytorch_grad_cam.utils.image.show_cam_on_image(image, activation, use_rgb=True)
            print(image_index, attribute)
            # save CAM activation
            celeba_dataset.save_cam(activation, overlay, attribute, image_index)
 

# Averages CAM images over the whole dataset, filtered by the given filter function
def average_cam(celeba_dataset, filter_function):
  for attribute in tqdm.tqdm(celeba_dataset.attributes):
    overlays = numpy.zeros(celeba_dataset.image_resolution, dtype=numpy.float64)
    activations = numpy.zeros(celeba_dataset.image_resolution[:-1], dtype=numpy.float64)
  
    # compute average over all images
    counter = 0
    for image_index in celeba_dataset:
      # apply filter function to get only the images that we want
      if filter_function(image_index,attribute):
        
        # load images
        print(image_index,end="\n")
        activation, overlay = celeba_dataset.load_cam(attribute, image_index)
        
        # compute average
        overlays += overlay
        activations += activation
        counter += 1
    # save averages
    celeba_dataset.save_cam(
      activations/counter,
      overlays/counter,
      attribute
    )


# Averages CAM images over the whole dataset, filtered by the given filter function
def average_perturb(celeba_dataset, filter_function):
  for attribute in tqdm.tqdm(celeba_dataset.attributes):
    overlays = torch.zeros([3,224,224])
    activations = numpy.zeros((224,224), dtype=numpy.float64)
  
    # compute average over all images
    counter = 0
    for image_index in celeba_dataset:
      # apply filter function to get only the images that we want
      if filter_function(image_index,attribute):
        # load images
        activation, overlay = celeba_dataset.load_perturb(attribute, image_index)
        
        # compute average
        overlays += overlay
        activations += activation
        counter += 1
    
    celeba_dataset.save_avg_perturb(
      activations/counter,
      overlays.numpy().transpose(1,2,0)/counter,
      attribute
    )
    
