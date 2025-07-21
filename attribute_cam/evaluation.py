import numpy
import tqdm
from PIL import Image

def accetptable_mask_ratio(activation, mask, mask_size):
  # count pixels with activation and check whether they are within or outside of mask
  activated = activation>0
  count_activated = numpy.sum(activated)

  if count_activated:
    count_true_positive = numpy.sum(activated * mask > 0)
#        count_false_positive = count_activated - count_true_positive

    amr = count_true_positive/count_activated
    amr_corr = (amr * ((mask.size-mask_size)/mask.size))
  else:
    amr = 0.
    amr_corr = 0.

#    mask_relative = round((mask_size/mask.size), 2)

  return amr, amr_corr #, mask_relative, count_activated, count_true_positive, count_false_positive


def save_activation_as_png(activation, filename):
    # Normalize activation to 0–255
    filename = f"../../../../local/scratch/chuber/Finalresults/unbalanced/corrRise_masks_black_3000_masks_50_patch/{filename}"
    activation_normalized = activation - numpy.min(activation)
    if numpy.max(activation_normalized) > 0:
        activation_normalized = activation_normalized / numpy.max(activation_normalized)
    activation_image = (activation_normalized * 255).astype(numpy.uint8)
    
    # Convert to Image and save
    img = Image.fromarray(activation_image)
    img.save(filename)


def proportional_energy(activation, mask, mask_size):
  # compute proportional energy
  # .. energy that is contained in the mask
  save_activation_as_png(mask,"maske.png")
  pos = numpy.sum(activation * (mask > 0))
  # .. total energy in the image
  total = numpy.sum(activation)

  if total:
    prop_energy = pos / total
    # corrected proportional energy
    prop_energy_corrected = (prop_energy * ((mask.size-mask_size)/mask.size))
    return prop_energy, prop_energy_corrected
  else:
    # no activation
    return 0., 0.


def statisics(celeba_dataset, filter_function, masks, mask_sizes, prop_energy=True):

  means = {}
  stds = {}
  for attribute in tqdm.tqdm(celeba_dataset.attributes):
    # compute AMR and corrected AMR for this attribute
    rates = []
    for image_index in celeba_dataset:
      if filter_function(image_index, attribute):
        activation, _ = celeba_dataset.load_cam(attribute, image_index)
        save_activation_as_png(activation,f"{attribute}{image_index}.png")
        if prop_energy:
          rates.append(proportional_energy(activation,masks[attribute],mask_sizes[attribute]))
        else:
          rates.append(accetptable_mask_ratio(activation,masks[attribute],mask_sizes[attribute]))

    # compute mean and std
    if rates:
      means[attribute] = numpy.mean(rates, axis=0)
      print(means)
      stds[attribute] = numpy.std(rates, axis=0)
    else:
      means[attribute] = [0,0]
      stds[attribute] = [0,0]

  return means, stds


# counts the number of positive and negative samples for each attribute
def class_counts(attributes, ground_truth, indexes, use_index = 0):
  rates = {attribute: [0,0,0] for attribute in attributes}
  for index, value in indexes.items():
    if value == use_index:
      for attribute in attributes:
        gt = int(ground_truth[index][attribute])
        rates[attribute][gt] += 1
  return {attribute:rate[1:3] for attribute,rate in rates.items()}


# computes the error rates for the given ground truth and predictions, averaged over the complete dataset, and separately for all attributes
def error_rate(celeba_dataset, ground_truth, prediction):
  errors = {}
  for attribute in celeba_dataset.attributes:
    # compute items of the confusion matrix: positives, negatives, false-positives and false-negatives
    FP,P,FN,N = 0,0,0,0
    for image_index in celeba_dataset:
      gt = ground_truth[image_index][attribute]
      pr = prediction[image_index][attribute]
      if gt == 1:
        P += 1
        if pr < 0:
          FN += 1
      else:
        N += 1
        if pr >= 0:
          FP += 1
    # compute false negative rate, false positive rate and (unbalanced) error rate
    errors[attribute] = (FN / P, FP / N, (FN + FP) / (P+N))
  return errors
