import numpy as np
from get_shifted_landmarks import get_shifted_landmarks_df, get_shifted_landmarks_img
import cv2

def get_masks(img_number, df):
    # get landmarks for image
    landmarks = get_shifted_landmarks_img(df, img_number)
    
    dict_masks = {}
        
    # forehead mask for attributes: 5
    mask_forehead = np.zeros((224, 224), np.uint8)
    middle_y_forehead = (landmarks[1] + landmarks[3])/2 - 27
    min_y_forehead = int(middle_y_forehead - 12)
    min_x_forehead = int(landmarks[0] - 15)
    max_y_forehead = int(middle_y_forehead + 15)
    max_x_forehead = int(landmarks[2] + 15)
    mask_forehead[min_y_forehead:max_y_forehead, min_x_forehead:max_x_forehead] = 255
    dict_masks[5] = mask_forehead
    
    # head_upper mask for attributes: 4, 28
    mask_head_upper = np.zeros((224, 224), np.uint8)
    y_hairline = (landmarks[1] + landmarks[3])/2 - 45
    middle_x_head = (landmarks[0] + landmarks[2])/2
    min_y_head_upper = int(y_hairline - 30)
    min_x_head_upper = int(middle_x_head - 55)
    max_y_head_upper = int(y_hairline + 15)
    max_x_head_upper = int(middle_x_head + 55)
    mask_head_upper[min_y_head_upper:max_y_head_upper, min_x_head_upper:max_x_head_upper] = 255
    dict_masks[4] = mask_head_upper
    dict_masks[28] = mask_head_upper
    
    # hat mask for attributes: 35
    mask_hat = np.zeros((224, 224), np.uint8)
    y_hairline = (landmarks[1] + landmarks[3])/2 - 45
    middle_x_head = (landmarks[0] + landmarks[2])/2
    min_y_hat = 5
    min_x_hat = int(middle_x_head - 70)
    max_y_hat = int(y_hairline + 15)
    max_x_hat = int(middle_x_head + 70)
    mask_hat[min_y_hat:max_y_hat, min_x_hat:max_x_hat] = 255
    dict_masks[35] = mask_hat
    
    # eye mask for attributes: 23
    mask_eye = np.zeros((224, 224), np.uint8)
    middle_y_eye = (landmarks[1] + landmarks[3])/2 
    min_y_eye = int(middle_y_eye - 10)
    min_x_eye = int(landmarks[0] - 13)
    max_y_eye = int(middle_y_eye + 10)
    max_x_eye = int(landmarks[2] + 13)
    mask_eye[min_y_eye:max_y_eye, min_x_eye:max_x_eye] = 255
    dict_masks[23] = mask_eye
    
    # eyeglasses mask for attributes: 15
    mask_eyeglasses = np.zeros((224, 224), np.uint8)
    min_y_eyeglasses = int(middle_y_eye - 15)
    min_x_eyeglasses = int(landmarks[0] - 22)
    max_y_eyeglasses = int(middle_y_eye + 15)
    max_x_eyeglasses = int(landmarks[2] + 22)
    mask_eyeglasses[min_y_eyeglasses:max_y_eyeglasses, min_x_eyeglasses:max_x_eyeglasses] = 255
    dict_masks[15] = mask_eyeglasses
    
    # eyebags mask for attributes: 3
    mask_eyebags = np.zeros((224, 224), np.uint8)
    min_y_eyebags = int(middle_y_eye + 4)
    min_x_eyebags = int(landmarks[0] - 13)
    max_y_eyebags = int(middle_y_eye + 15)
    max_x_eyebags = int(landmarks[2] + 13)
    mask_eyebags[min_y_eyebags:max_y_eyebags, min_x_eyebags:max_x_eyebags] = 255
    dict_masks[3] = mask_eyebags
    
    # eyebrows mask for attributes: 12
    mask_eyebrows = np.zeros((224, 224), np.uint8)
    min_y_eyebrows = int(middle_y_eye - 18)
    min_x_eyebrows = int(landmarks[0] - 13)
    max_y_eyebrows = int(middle_y_eye - 2)
    max_x_eyebrows = int(landmarks[2] + 13)
    mask_eyebrows[min_y_eyebrows:max_y_eyebrows, min_x_eyebrows:max_x_eyebrows] = 255
    dict_masks[1] = mask_eyebrows
    dict_masks[12] = mask_eyebrows
    
    # nose mask for attributes: 7, 27
    mask_nose = np.zeros((224, 224), np.uint8)
    middle_y_nose = landmarks[5]
    min_y_nose = int(middle_y_nose - 15)
    min_x_nose = int(landmarks[4] - 15)
    max_y_nose = int(middle_y_nose + 8)
    max_x_nose = int(landmarks[4] + 15)
    mask_nose[min_y_nose:max_y_nose, min_x_nose:max_x_nose] = 255
    dict_masks[7] = mask_nose
    dict_masks[27] = mask_nose
    
    # cheeks mask for attributes: 19, 29
    # cheek_l
    mask_cheek_l = np.zeros((224, 224), np.uint8)
    min_y_cheek_l = int(middle_y_eye + 4)
    min_x_cheek_l = int(landmarks[0] - 25)
    max_y_cheek_l = int(landmarks[5] + 5)
    max_x_cheek_l = int(landmarks[4])
    mask_cheek_l[min_y_cheek_l:max_y_cheek_l, min_x_cheek_l:max_x_cheek_l] = 255
    # cheek_r
    mask_cheek_r = np.zeros((224, 224), np.uint8)
    min_y_cheek_r = int(middle_y_eye + 4)
    min_x_cheek_r = int(landmarks[4])
    max_y_cheek_r = int(landmarks[5] + 5)
    max_x_cheek_r = int(landmarks[2] + 25)
    mask_cheek_r[min_y_cheek_r:max_y_cheek_r, min_x_cheek_r:max_x_cheek_r] = 255
    mask_cheeks = np.add(mask_cheek_l, mask_cheek_r)
    dict_masks[19] = mask_cheeks
    dict_masks[29] = mask_cheeks
    
    # sideburns mask for attributes: 30
    # sideburn_l
    mask_sideburn_l = np.zeros((224, 224), np.uint8)
    min_y_sideburn_l = int(middle_y_eye + 4)
    min_x_sideburn_l = int(landmarks[0] - 25)
    max_y_sideburn_l = int(landmarks[5] + 28)
    max_x_sideburn_l = int(landmarks[0] - 5)
    mask_sideburn_l[min_y_sideburn_l:max_y_sideburn_l, min_x_sideburn_l:max_x_sideburn_l] = 255
    # sideburn_r
    mask_sideburn_r = np.zeros((224, 224), np.uint8)
    min_y_sideburn_r = int(middle_y_eye + 4)
    min_x_sideburn_r = int(landmarks[2] + 5)
    max_y_sideburn_r = int(landmarks[5] + 28)
    max_x_sideburn_r = int(landmarks[2] + 25)
    mask_sideburn_r[min_y_sideburn_r:max_y_sideburn_r, min_x_sideburn_r:max_x_sideburn_r] = 255
    dict_masks[30] = np.add(mask_sideburn_l, mask_sideburn_r)
    
    # earrings mask for attributes: 34   
    # earring_l
    mask_earring_l = np.zeros((224, 224), np.uint8)
    min_y_earring_l = int(middle_y_nose - 15)
    min_x_earring_l = int(landmarks[0] - 34)
    max_y_earring_l = int(middle_y_nose + 42)
    max_x_earring_l = int(landmarks[0] - 12)
    mask_earring_l[min_y_earring_l:max_y_earring_l, min_x_earring_l:max_x_earring_l] = 255
    # earring_r
    mask_earring_r = np.zeros((224, 224), np.uint8)
    min_y_earring_r = int(middle_y_nose - 15)
    min_x_earring_r = int(landmarks[2] + 12)
    max_y_earring_r = int(middle_y_nose + 42)
    max_x_earring_r = int(landmarks[2] + 34)
    mask_earring_r[min_y_earring_r:max_y_earring_r, min_x_earring_r:max_x_earring_r] = 255
    dict_masks[34] = np.add(mask_earring_l, mask_earring_r)
    
    # mouth mask for attributes: 6, 21, 31, 36
    mask_mouth = np.zeros((224, 224), np.uint8)
    middle_y_mouth = (landmarks[7] + landmarks[9])/2 
    min_y_mouth = int(middle_y_mouth - 10)
    min_x_mouth = int(landmarks[6] - 6)
    max_y_mouth = int(middle_y_mouth + 15)
    max_x_mouth = int(landmarks[8] + 6)
    mask_mouth[min_y_mouth:max_y_mouth, min_x_mouth:max_x_mouth] = 255
    dict_masks[6] = mask_mouth
    dict_masks[21] = mask_mouth
    dict_masks[31] = mask_mouth
    dict_masks[36] = mask_mouth
    
    # mustache mask for attributes: 22
    mask_mustache = np.zeros((224, 224), np.uint8)
    min_y_mustache = int(middle_y_mouth -  15)
    min_x_mustache = int(landmarks[6] - 10)
    max_y_mustache = int(middle_y_mouth)
    max_x_mustache = int(landmarks[8] + 10)
    mask_mustache[min_y_mustache:max_y_mustache, min_x_mustache:max_x_mustache] = 255
    dict_masks[22] = mask_mustache
    
    # beard mask for attributes: 24
    mask_beard = np.zeros((224, 224), np.uint8)
    min_y_beard = int(middle_y_eye + 4)
    min_x_beard = int(landmarks[6] - 25)
    max_y_beard = int(middle_y_mouth + 32)
    max_x_beard = int(landmarks[8] + 25)
    mask_beard[min_y_beard:max_y_beard, min_x_beard:max_x_beard] = 255
    dict_masks[24] = mask_beard
    
    # goatee mask for attributes: 16
    mask_goatee = np.zeros((224, 224), np.uint8)
    min_y_goatee = int(middle_y_mouth)
    min_x_goatee = int(landmarks[6])
    max_y_goatee = int(middle_y_mouth + 30)
    max_x_goatee = int(landmarks[8])
    mask_goatee[min_y_goatee:max_y_goatee, min_x_goatee:max_x_goatee] = 255
    dict_masks[16] = mask_goatee
    
    # doublechin mask for attributes: 14
    mask_doublechin = np.zeros((224, 224), np.uint8)
    min_y_doublechin = int(middle_y_mouth + 30)
    min_x_doublechin = int(landmarks[6] - 6)
    max_y_doublechin = int(middle_y_mouth + 60)
    max_x_doublechin = int(landmarks[8] + 6)
    mask_doublechin[min_y_doublechin:max_y_doublechin, min_x_doublechin:max_x_doublechin] = 255
    dict_masks[14] = mask_doublechin
    
    # neck mask for attributes: 37, 38
    mask_neck = np.zeros((224, 224), np.uint8)
    min_y_neck = int(middle_y_mouth + 30)
    min_x_neck = int(landmarks[6] - 30)
    max_y_neck = 224
    max_x_neck = int(landmarks[8] + 30)
    mask_neck[min_y_neck:max_y_neck, min_x_neck:max_x_neck] = 255
    dict_masks[37] = mask_neck
    dict_masks[38] = mask_neck
   
    # face mask (without hair) for attributes: 18, 25, 26
    mask_face = np.zeros((224, 224), np.uint8)
    min_y_face = int(middle_y_eye - 45)
    min_x_face = int(landmarks[4] - 50)
    max_y_face = int(middle_y_mouth + 35)
    max_x_face = int(landmarks[4] + 55)
    mask_face[min_y_face:max_y_face, min_x_face:max_x_face] = 255
    dict_masks[18] = mask_face
    dict_masks[25] = mask_face
    dict_masks[26] = mask_face
    
    # global mask for attributes: 2, 10, 13, 20, 39
    mask_global = np.zeros((224, 224), np.uint8)
    min_y_global = int(middle_y_nose - 100)
    min_x_global = int(landmarks[4] - 80)
    max_y_global = int(224)
    max_x_global = int(landmarks[4] + 75)
    mask_global[min_y_global:max_y_global, min_x_global:max_x_global] = 255
    dict_masks[2] = mask_global
    dict_masks[10] = mask_global
    dict_masks[13] = mask_global
    dict_masks[20] = mask_global
    dict_masks[39] = mask_global
    
    # hair mask (without face) for attributes: 8, 9, 11, 17, 32, 33
    mask_hair = np.zeros((224, 224), np.uint8)
    img = np.zeros((224, 224), np.uint8)
    
    center_coordinates = (landmarks[4], landmarks[5] - 15)
    axesLength = (45, 60)
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (255, 255, 255)
    thickness = -1
    img_mask_face = cv2.ellipse(img, center_coordinates, axesLength,
               angle, startAngle, endAngle, color, thickness)
    np.reshape(img_mask_face, (224, 224))
    
    mask_hair = mask_global - img_mask_face
    dict_masks[8] = mask_hair
    dict_masks[9] = mask_hair
    dict_masks[11] = mask_hair
    dict_masks[17] = mask_hair
    dict_masks[32] = mask_hair
    dict_masks[33] = mask_hair
    
    # 5 o clock shadow mask for attributes: 0
    dict_masks[0] = mask_beard
    
    myKeys = list(dict_masks.keys())
    myKeys.sort()
    sorted_dict_masks = {i: dict_masks[i] for i in myKeys}
    
    return sorted_dict_masks   
