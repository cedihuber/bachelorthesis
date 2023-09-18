import numpy
import tqdm

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


def proportional_energy(activation, mask, mask_size):
  # compute proportional energy
  # .. energy that is contained in the mask
  pos = numpy.sum(activation * (mask > 0))
  # .. total energy in the image
  total = numpy.sum(activation)

  prop_energy = pos / total
  # corrected proportional energy
  prop_energy_corrected = (prop_energy * ((mask.size-mask_size)/mask.size))
  return prop_energy, prop_energy_corrected


def amr_statisics(celeba_dataset, filter_function, masks, mask_sizes):

  means = {}
  stds = {}
  for attribute in tqdm.tqdm(celeba_dataset.attributes):
    # compute AMR and corrected AMR for this attribute
    amr = []
    for image_index in celeba_dataset:
      if filter_function(image_index, attribute):
        activation, _ = celeba_dataset.load_cam(attribute, image_index)

        amr.append(accetptable_mask_ratio(activation,masks[attribute],mask_sizes[attribute]))

    # compute mean and std
    if amr:
      means[attribute] = numpy.mean(amr, axis=0)
      stds[attribute] = numpy.std(amr, axis=0)
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
