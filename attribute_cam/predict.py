import torch
import torchvision
from importlib.machinery import SourceFileLoader

def load_model(model_type):
    MainModel = SourceFileLoader("MainModel", "attribute_cam/AFFACT.py").load_module()
    network = torch.load("attribute_cam/AFFACT_" + model_type + ".pth")
    network.identity = torch.nn.Identity()

    # setup network
    network.eval()
    # OPTIONAL: set to cuda environment if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    return network, device

def preprocess_img(img_path, device):
    # load example image (already preprocessed)
    image = torchvision.io.image.read_image(img_path)
    # convert to the required data type
    image = image / 255.
    # add the required batch dimension
    tensor = image.unsqueeze(0)
    # and put the data to device (optional)
    tensor = tensor.to(device)

    return tensor

def predict(network, tensor):
    with torch.no_grad():
      # extract feature vector
      attributes = network(tensor)
      # transform it into 1D numpy array
      attributes = attributes.cpu().numpy().flatten()

    return attributes
