import numpy as np

from .dataset import ATTRIBUTES

# Creates masks for each attribute where we expect activation
def get_masks():

    dict_masks = {}
    dict_sizes = {}

    # upperhead mask for attributes: 4, 28
    mask_upperhead = np.zeros((224, 224), np.uint8)
    mask_upperhead[0*32:3*32, 2*32:5*32] = 255
    dict_masks[4] = mask_upperhead
    dict_masks[28] = mask_upperhead
    # size
    size_mask_upperhead = (3*32-0*32) * (5*32-2*32)
    dict_sizes[4] = size_mask_upperhead
    dict_sizes[28] = size_mask_upperhead

    # hat mask for attributes: 35
    mask_hat = np.zeros((224, 224), np.uint8)
    mask_hat[0*32:3*32, 1*32:6*32] = 255
    dict_masks[35] = mask_hat
    # size
    size_mask_hat = (3*32-0*32) * (6*32-1*32)
    dict_sizes[35] = size_mask_hat

    # eyes mask for attributes: 1, 3, 5, 12, 15, 23
    mask_eyes = np.zeros((224, 224), np.uint8)
    mask_eyes[2*32:4*32, 2*32:5*32] = 255
    dict_masks[1] = mask_eyes
    dict_masks[3] = mask_eyes
    dict_masks[5] = mask_eyes
    dict_masks[12] = mask_eyes
    dict_masks[15] = mask_eyes
    dict_masks[23] = mask_eyes
    # size
    size_mask_eyes = (4*32-2*32) * (5*32-2*32)
    dict_sizes[1] = size_mask_eyes
    dict_sizes[3] = size_mask_eyes
    dict_sizes[5] = size_mask_eyes
    dict_sizes[12] = size_mask_eyes
    dict_sizes[15] = size_mask_eyes
    dict_sizes[23] = size_mask_eyes

    # nose_cheeks mask for attributes: 0, 7, 18, 19, 24, 27, 29
    mask_nose_cheeks = np.zeros((224, 224), np.uint8)
    mask_nose_cheeks[3*32:5*32, 2*32:5*32] = 255
    dict_masks[0] = mask_nose_cheeks
    dict_masks[7] = mask_nose_cheeks
    dict_masks[18] = mask_nose_cheeks
    dict_masks[19] = mask_nose_cheeks
    dict_masks[24] = mask_nose_cheeks
    dict_masks[27] = mask_nose_cheeks
    dict_masks[29] = mask_nose_cheeks
    # size
    size_mask_nose_cheeks = (5*32-3*32) * (5*32-2*32)
    dict_sizes[0] = size_mask_nose_cheeks
    dict_sizes[7] = size_mask_nose_cheeks
    dict_sizes[18] = size_mask_nose_cheeks
    dict_sizes[19] = size_mask_nose_cheeks
    dict_sizes[24] = size_mask_nose_cheeks
    dict_sizes[27] = size_mask_nose_cheeks
    dict_sizes[29] = size_mask_nose_cheeks


    # ears mask for attributes: 30, 34
    mask_ear_l = np.zeros((224, 224), np.uint8)
    mask_ear_r = np.zeros((224, 224), np.uint8)
    mask_ear_l[3*32:6*32, 1*32:3*32] = 255
    mask_ear_r[3*32:6*32, 4*32:6*32] = 255
    mask_ears = np.add(mask_ear_l, mask_ear_r)
    dict_masks[34] = mask_ears
    dict_masks[30] = mask_ears
    # size
    size_mask_ear_l = (6*32-3*32) * (3*32-1*32)
    size_mask_ear_r = (6*32-3*32) * (6*32-4*32)
    size_mask_ears = size_mask_ear_l + size_mask_ear_r
    dict_sizes[34] = size_mask_ears
    dict_sizes[30] = size_mask_ears


    # mouth mask for attributes: 6, 21, 22, 31, 36
    mask_mouth = np.zeros((224, 224), np.uint8)
    mask_mouth[4*32:5*32, 2*32:5*32] = 255
    dict_masks[6] = mask_mouth
    dict_masks[21] = mask_mouth
    dict_masks[22] = mask_mouth
    dict_masks[31] = mask_mouth
    dict_masks[36] = mask_mouth
    # size
    size_mask_mouth = (5*32-4*32) * (5*32-2*32)
    dict_sizes[6] = size_mask_mouth
    dict_sizes[21] = size_mask_mouth
    dict_sizes[22] = size_mask_mouth
    dict_sizes[31] = size_mask_mouth
    dict_sizes[36] = size_mask_mouth


    # goatee mask for attributes: 16
    mask_goatee = np.zeros((224, 224), np.uint8)
    mask_goatee[4*32:6*32, 3*32:4*32] = 255
    dict_masks[16] = mask_goatee
    # size
    size_mask_goatee = (6*32-4*32) * (4*32-3*32)
    dict_sizes[16] = size_mask_goatee

    # doublechin mask for attributes: 14
    mask_doublechin = np.zeros((224, 224), np.uint8)
    mask_doublechin[5*32:6*32, 2*32:5*32] = 255
    dict_masks[14] = mask_doublechin
    # size
    size_mask_doublechin = (6*32-5*32) * (5*32-2*32)
    dict_sizes[14] = size_mask_doublechin

    # neck mask for attributes: 37, 38
    mask_neck = np.zeros((224, 224), np.uint8)
    mask_neck[5*32:7*32, 2*32:5*32] = 255
    dict_masks[37] = mask_neck
    dict_masks[38] = mask_neck
    # size
    size_mask_neck = (7*32-5*32) * (5*32-2*32)
    dict_sizes[37] = size_mask_neck
    dict_sizes[38] = size_mask_neck

    # face mask (without hair) for attributes: 25, 26
    mask_face = np.zeros((224, 224), np.uint8)
    mask_face[1*32:6*32, 2*32:5*32] = 255
    dict_masks[25] = mask_face
    dict_masks[26] = mask_face
    # size
    size_mask_face = (6*32-1*32) * (5*32-2*32)
    dict_sizes[25] = size_mask_face
    dict_sizes[26] = size_mask_face

    # global mask for attributes: 2, 10, 13, 20, 39
    mask_global = np.zeros((224, 224), np.uint8)
    mask_global[1*32:7*32, 1*32:6*32] = 255
    dict_masks[2] = mask_global
    dict_masks[10] = mask_global
    dict_masks[13] = mask_global
    dict_masks[20] = mask_global
    dict_masks[39] = mask_global
    # size
    size_mask_global = (7*32-1*32) * (6*32-1*32)
    dict_sizes[2] = size_mask_global
    dict_sizes[10] = size_mask_global
    dict_sizes[13] = size_mask_global
    dict_sizes[20] = size_mask_global
    dict_sizes[39] = size_mask_global

    # hair mask for attributes: 8, 9, 11, 17, 32, 33
    mask_face2 = np.zeros((224, 224), np.uint8)
    mask_face2[2*32:7*32, 2*32:5*32] = 255
    mask_hair = mask_global-mask_face2
    dict_masks[8] = mask_hair
    dict_masks[9] = mask_hair
    dict_masks[11] = mask_hair
    dict_masks[17] = mask_hair
    dict_masks[32] = mask_hair
    dict_masks[33] = mask_hair
    # size
    size_mask_face2 = (7*32-2*32) * (5*32-2*32)
    size_mask_hair = size_mask_global-size_mask_face2
    dict_sizes[8] = size_mask_hair
    dict_sizes[9] = size_mask_hair
    dict_sizes[11] = size_mask_hair
    dict_sizes[17] = size_mask_hair
    dict_sizes[32] = size_mask_hair
    dict_sizes[33] = size_mask_hair

    # use attribute names instead if indexes
    masks = {ATTRIBUTES[i]: mask for i,mask in dict_masks.items()}
    sizes = {ATTRIBUTES[i]: size for i,size in dict_sizes.items()}

    return masks, sizes
