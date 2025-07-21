import os
import csv

from .dataset import ATTRIBUTES

def prediction_file(output_directory, which_set, model_type):
  return os.path.join(output_directory, f"Prediction-{which_set}-{model_type}.csv")

# reads CSV lists from file, including ground truth or predictions
# For each image, it stores a dictionary containing attribute as keys and gt/prediction and value
def read_list(list_file, delimiter, header_rows, split_attributes=True):
  result = {}
  with open(list_file, "r") as r:
    reader = csv.reader(r, delimiter=delimiter, skipinitialspace=True)
    # skip header
    for _ in range(header_rows):
      next(reader)

    # read values, convert to float and assign attribute
    for splits in reader:
      if split_attributes:
        assert len(splits) == len(ATTRIBUTES)+1
        # store values as dictionary
        result[os.path.splitext(splits[0])[0]] = {
          attribute : float(splits[i+1]) for i, attribute in enumerate(ATTRIBUTES)
        }
      else:
        result[os.path.splitext(splits[0])[0]] = int(splits[1])


  return result

# We define several filter functions based on the ground truth and the prediction
FILTERS={
  "none" : lambda ground_truth,prediction: True,
  "gt=1" : lambda ground_truth,prediction: ground_truth == 1,
  "gt=-1" : lambda ground_truth,prediction: ground_truth == -1,
  "pr=1" : lambda ground_truth,prediction: prediction >= 0,
  "pr=-1" : lambda ground_truth,prediction: prediction < 0,
  "gt==pr" : lambda ground_truth,prediction: ground_truth * prediction >= 0,
  "gt!=pr" : lambda ground_truth,prediction: ground_truth * prediction < 0,
  "majority" : lambda majority, prediction: majority * prediction >= 0,
  "minority" : lambda majority, prediction: majority * prediction < 0,
  "majority_p" : lambda majority, prediction, ground_truth: majority * prediction >= 0 and ground_truth * prediction >= 0,
  "minority_p" : lambda majority, prediction, ground_truth: majority * prediction < 0 and ground_truth * prediction >= 0,
  "majority_n" : lambda majority, prediction, ground_truth: majority * prediction >= 0 and ground_truth * prediction < 0,
  "minority_n" : lambda majority, prediction, ground_truth: majority * prediction < 0 and ground_truth * prediction < 0,
}

# This class will apply the given filter for specific files from the dataset
class Filter1:
  def __init__(self, ground_truth, prediction, majority, filter = "none"):
    self.ground_truth = ground_truth
    self.prediction = prediction
    self.majority = majority
    self.filter_function = FILTERS[filter]
    self.test = FILTERS["pr=1"]
    self.filter = filter
    
  def __call__(self, item, attribute):
    # apply the filter for the given image and attribute
    #print("hs")
    if self.filter == "majority" or self.filter == "minority":
        # print(f"{attribute}, {self.majority[attribute]}, {self.prediction[item][attribute]}")
        # if(self.filter_function(self.majority[attribute], self.prediction[item][attribute]) == self.test(self.ground_truth[item][attribute], self.prediction[item][attribute])):
        #   print("they are the same!!!!!!!!")
        return self.filter_function(self.majority[attribute], self.prediction[item][attribute])
    elif self.filter == "majority_p" or self.filter == "minority_p" or self.filter == "majority_n" or self.filter == "minority_n":
      return self.filter_function(self.majority[attribute], self.prediction[item][attribute], self.ground_truth[item][attribute])
    else:
        return self.filter_function(self.ground_truth[item][attribute], self.prediction[item][attribute])
