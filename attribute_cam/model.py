import torch
import tqdm
from importlib.machinery import SourceFileLoader
import pkg_resources

class AFFACT:
  def __init__(self, model_type, device):
    # loads the model
    #model_file = pkg_resources.resource_filename(__name__, "..", "model", "AFFACT.py")
    #weight_file = pkg_resources.resource_filename(__name__, "..", "model", f"AFFACT_{model_type}.pth")
    model_file = "/home/user/chuber/attribute-cam/model/AFFACT.py"
    weight_file = f"/home/user/chuber/attribute-cam/model/AFFACT_{model_type}.pth"
    MainModel = SourceFileLoader("MainModel", model_file).load_module()
    network = torch.load(weight_file)
    # we have to add the Identity layer afterward since the original weights do not include it
    network.identity = torch.nn.Identity()
    # setup network
    network.eval()
    self.network = network.to(device)
    self.device = device

  def predict(self, tensor):
    with torch.no_grad():
      # extract feature vector
      attributes = self.network(tensor.to(self.device))
      # transform it into 1D numpy array
      return attributes.cpu().numpy().flatten()

  def model(self):
    return self.network

  def cam_target_layers(self):
    return [self.network.identity]

  # runs a prediction for all images of the dataset and all attributes, and writes them to file
  def predict_all(self, celeba_dataset, output_file):
    with open(output_file, "w") as w:
      for item in tqdm.tqdm(celeba_dataset):
        # predict attribute
        #add attribute name 
        prediction = self.predict(celeba_dataset.source_tensor(item))
        w.write(item+",")
        w.write(",".join([f"{value:+1.4f}" for value in prediction]))
        w.write("\n")