import torch
import tqdm
from importlib.machinery import SourceFileLoader
import pkg_resources

class AFFACT:
  def __init__(self, model_type, device):
    model_file = pkg_resources.resource_filename(__name__, "AFFACT.py")
    weight_file = pkg_resources.resource_filename(__name__, f"AFFACT_{model_type}.pth")
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

  def predict_all(self, celeba_dataset, output_file):
    with open(output_file, "w") as w:
      for item in tqdm.tqdm(celeba_dataset):
        prediction = self.predict(celeba_dataset.source_tensor(item))
        w.write(item+",")
        w.write(",".join([f"{value:+1.4f}" for value in prediction]))
        w.write("\n")
