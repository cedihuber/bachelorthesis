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

  def predict_rise(self, tensor):
    with torch.no_grad():
      # extract feature vector
      attributes = self.network(tensor.to(self.device))
      # transform it into 1D numpy array
      return attributes.flatten() 


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
        #mask = celeba_dataset.source_tensor(item)
        #print(f"Image shape: {mask.shape}, Min: {mask.min()}, Max: {mask.max()}")
        #print(f'min{celeba_dataset.source_tensor(item).min()}')
        #print(f'max{celeba_dataset.source_tensor(item).max()}')
        print(celeba_dataset.source_tensor(item).shape)
        prediction = self.predict(celeba_dataset.source_tensor(item))
        w.write(item+",")
        w.write(",".join([f"{value:+1.4f}" for value in prediction]))
        w.write("\n")
        
     
  def predict_perturbed(self, perturbed_images, output_file):
    
    images, filenames = perturbed_images
    num_images = images.shape[0]
    batch_size = 20
    all_predictions = []
    
    images = images.to(self.device)
    with torch.no_grad():
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)  
        
        # Apply sigmoid activation
        #nicht in ein file schreiben aber returnen und dann direkt für die erstellung von saliency map verwenden
        predictions = torch.sigmoid(predictions)
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
        all_predictions.append(grouped_predictions)
        
      final_predictions = torch.cat(all_predictions, dim=0)
      return final_predictions
  
    
  def predict_logit(self, perturbed_images):
    
    images, filenames = perturbed_images
    num_images = images.shape[0]
    batch_size = 20
    all_predictions = []
    
    #images = images.to(self.device)
    with torch.no_grad():
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx] #shape (batch_size, 3, 224, 224)
        #print(batch_images.shape)
        #print(f'min{batch_images.min()}')
        #print(f'max{batch_images.max()}')
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)  

        # Apply sigmoid activation
        #nicht in ein file schreiben aber returnen und dann direkt für die erstellung von saliency map verwenden
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
        all_predictions.append(grouped_predictions)

      final_predictions = torch.cat(all_predictions, dim=0)
      return final_predictions 
    
    
    
  def predict_logit_absolute(self, perturbed_images, output_file):
  
    images, filenames = perturbed_images
    num_images = images.shape[0]
    batch_size = 2
    all_predictions = []
   #images = images.to(self.device)
    with torch.no_grad():
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx] #shape (batch_size, 3, 224, 224)
        #print(batch_images.shape)
        #print(f'min{batch_images.min()}')
        #print(f'max{batch_images.max()}')
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        predictions = torch.abs(predictions)
       # Apply sigmoid activation
        #nicht in ein file schreiben aber returnen und dann direkt für die erstellung von saliency map verwenden
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
        all_predictions.append(grouped_predictions)
      
      final_predictions = torch.cat(all_predictions, dim=0)
      return final_predictions  
  
          
  def predict_file_logit(self, perturbed_images, output_file):
    images, filenames = perturbed_images
    num_images = images.shape[0]
    batch_size = 20
    with open(output_file, "a") as w:
      #print(images.shape)
      #print(filenames)
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        predictions = torch.sigmoid(predictions)
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
       
        for prediction, filename in tqdm.tqdm(zip(grouped_predictions, filenames[start_idx:end_idx])):
          w.write(f"{filename},")
          w.write(",".join([f"{value:+1.4f}" for value in prediction]))
          w.write("\n")
          
          
          
          
  def predict_corrrise(self, perturbed_images):
    
    images, filenames = perturbed_images
    num_images = images.shape[0]
    batch_size = 40
    all_predictions = []
    
    images = images.to(self.device)
    with torch.no_grad():
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)  

        # Apply sigmoid activation
        #nicht in ein file schreiben aber returnen und dann direkt für die erstellung von saliency map verwenden
        predictions = torch.sigmoid(predictions)
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
        all_predictions.append(grouped_predictions)
      
      final_predictions = torch.cat(all_predictions, dim=0) #shape 500, 40
    
      return final_predictions
    
      
  def predict_corrrise_batches(self, perturbed_images):
    
    images, filenames = perturbed_images # images shape(155,3,224,224)
    #print(images.shape)
    num_images = images.shape[0]
    print(f'num_images{num_images}')
    batch_size = 4
    res = []
    images = images.to(self.device)
    with torch.no_grad():
      all_predictions = []
      for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]
        predictions = self.predict(batch_images)
        predictions = torch.tensor(predictions, dtype=torch.float32)  
        # Apply sigmoid activation
        #nicht in ein file schreiben aber returnen und dann direkt für die erstellung von saliency map verwenden
        predictions = torch.sigmoid(predictions)
        num_attributes = 40
        grouped_predictions = predictions.view(-1, num_attributes)
        all_predictions.append(grouped_predictions)
    
      final_predictions = torch.cat(all_predictions, dim=0) #shape 500, 40
      print(final_predictions.shape)
      return final_predictions