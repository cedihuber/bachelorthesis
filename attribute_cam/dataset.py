import os
import collections
import torchvision
import torch
import numpy

ATTRIBUTES=[
  '5_o_Clock_Shadow',
  'Arched_Eyebrows',
  'Attractive',
  'Bags_Under_Eyes',
  'Bald',
  'Bangs',
  'Big_Lips',
  'Big_Nose',
  'Black_Hair',
  'Blond_Hair',
  'Blurry',
  'Brown_Hair',
  'Bushy_Eyebrows',
  'Chubby',
  'Double_Chin',
  'Eyeglasses',
  'Goatee',
  'Gray_Hair',
  'Heavy_Makeup',
  'High_Cheekbones',
  'Male',
  'Mouth_Slightly_Open',
  'Mustache',
  'Narrow_Eyes',
  'No_Beard',
  'Oval_Face',
  'Pale_Skin',
  'Pointy_Nose',
  'Receding_Hairline',
  'Rosy_Cheeks',
  'Sideburns',
  'Smiling',
  'Straight_Hair',
  'Wavy_Hair',
  'Wearing_Earrings',
  'Wearing_Hat',
  'Wearing_Lipstick',
  'Wearing_Necklace',
  'Wearing_Necktie',
  'Young'
]

class CelebA:
  def __init__(self, filtered_lists, source_directory, cam_directory=None, number_of_images=None, attributes=None, filter_type="none", extension=".png", image_resolution=(224,224,3)):

    # get default values for None parameters
    number_of_images = number_of_images or 100000
    attributes = attributes or ATTRIBUTES

    self.source_directory = source_directory
    self.cam_directory = cam_directory
    self.extension = extension
    self.image_resolution = image_resolution
    self.attributes = {attribute : ATTRIBUTES.index(attribute) for attribute in attributes}
    self.filter_type=filter_type

    # load image lists
    self.images = [os.path.splitext(i)[0] for filtered_list in filtered_lists for n,i in enumerate(open(filtered_list, 'r'))  if i and n < number_of_images]

  def __iter__(self):
    self.queue = collections.deque(self.images)
    return self

  def __next__(self):
    if self.queue:
      return self.queue.pop()
    raise StopIteration

  def __len__(self):
    return len(self.images)

  def source_filename(self, item):
    return os.path.join(self.source_directory, item + self.extension)

  def cam_filename(self, attribute, item = None):
    if item is None:
      return os.path.join(self.cam_directory, self.filter_type, attribute + self.extension)
    return os.path.join(self.cam_directory, attribute, item + self.extension)

  def source_tensor(self, item):
    # load image (already preprocessed)
    image = torchvision.io.image.read_image(self.source_filename(item))
    # convert to the required data type
    image = image / 255.
    # add the required batch dimension
    return image.unsqueeze(0)

  def source_image(self, item):
    # load tensor
    tensor = self.source_tensor(item)
    # convert to CV2-style image
    return tensor, tensor[0].numpy().transpose(1, 2, 0)

  def save_cam(self, activation, overlay, attribute, item=None):
    filename = self.cam_filename(attribute, item)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_png(torch.tensor(overlay.transpose(2,0,1), dtype=torch.uint8), filename)
    numpy.save(filename+".npy", activation)

  def load_cam(self, attribute, item=None):
    filename = self.cam_filename(attribute, item)
    overlay = torchvision.io.image.read_image(filename).numpy().transpose(1,2,0)
    activation = numpy.load(filename + ".npy")
    return activation, overlay
