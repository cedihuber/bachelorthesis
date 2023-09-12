import os
import csv

from .dataset import ATTRIBUTES

def prediction_file(output_directory, which_set, model_type):
  return os.path.join(output_directory, f"Prediction-{which_set}-{model_type}.csv")

def read_list(list_file, delimiter, header_rows):
  result = {}
  with open(list_file, "r") as r:
    reader = csv.reader(r, delimiter=delimiter, skipinitialspace=True)
    # skip header
    for _ in range(header_rows):
      next(reader)

    # read values, convert to float and assign attribute
    for splits in reader:
      assert len(splits) == len(ATTRIBUTES)+1, f"{len(splits)} != {len(ATTRIBUTES)+1}"
      result[os.path.splitext(splits[0])[0]] = {
        attribute : float(splits[i+1]) for i, attribute in enumerate(ATTRIBUTES)
      }

  return result


FILTERS={
  "none" : lambda ground_truth,prediction: True,
  "gt=1" : lambda ground_truth,prediction: ground_truth == 1,
  "gt=-1" : lambda ground_truth,prediction: ground_truth == -1,
  "pr=1" : lambda ground_truth,prediction: prediction >= 0,
  "pr=-1" : lambda ground_truth,prediction: prediction < 0,
  "gt==pr" : lambda ground_truth,prediction: ground_truth * prediction >= 0,
  "gt!=pr" : lambda ground_truth,prediction: ground_truth * prediction < 0,
}

class Filter:
  def __init__(self, ground_truth, prediction, filter = "none"):
    self.ground_truth = ground_truth
    self.prediction = prediction
    self.filter_function = FILTERS[filter]

  def __call__(self, item, attribute):
    return self.filter_function(self.ground_truth[item][attribute], self.prediction[item][attribute])
