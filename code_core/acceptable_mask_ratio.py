import os
import cv2
import numpy as np

def get_accetptable_mask_ratio(img_path, img_nr, output_dir, attribute_name, mask, mask_size):
    # get class activation map
    cam_name = f'{img_nr}_{attribute_name}_cam.jpg'
    DIR_cam = os.path.join(output_dir, cam_name)
    cam_bgr = cv2.imread(DIR_cam, cv2.IMREAD_COLOR)
    
    cam_on_img_name = f'{img_nr}_{attribute_name}_cam_on_img.jpg'
    DIR_cam_on_img = os.path.join(output_dir, cam_on_img_name)
    cam_on_img = cv2.imread(DIR_cam_on_img, cv2.IMREAD_COLOR)
    
    grayscale_cam_name = f'{img_nr}_{attribute_name}_grayscale.jpg'
    DIR_grayscale_cam = os.path.join(output_dir, grayscale_cam_name)
    grayscale_cam = cv2.imread(DIR_grayscale_cam, cv2.IMREAD_GRAYSCALE)
    
    # count pixels with activation and check whether they are within or outside of mask
    discriminative_area = np.zeros((224, 224), np.uint8)
    count_activated = 0
    count_true_positive = 0
    count_false_positive = 0
    #count_mask = 0
    for line in range(224):
        for pxl in range(224):
            if (grayscale_cam[line][pxl] > 0): # there is any activation 
                # make pixel white (those without activity will be black)
                discriminative_area[line][pxl] = 255
                count_activated += 1
                # check if activated pixel is within mask --> positive true
                if (mask[line][pxl] == 255):
                    count_true_positive += 1
                else: # false positive
                    count_false_positive += 1  
                    
    # overlay image and mask
    masked_img = cv2.bitwise_and(cam_on_img, cam_on_img, mask = mask)
    # overlay class activation map and mask
    masked_img_activation = cv2.bitwise_and(cam_bgr, cam_bgr, mask = discriminative_area)
    
    # compute acceptable mask ratio
    try:
        amr = count_true_positive/count_activated
        amr_corr = (amr * ((50176-mask_size)/50176))
    except ZeroDivisionError:
        amr = 0.0
        amr_corr= 0.0
    
    mask_relative = round((mask_size/50176), 2)
    
    # save imgs
    cv2.imwrite(os.path.join(output_dir, f'{img_nr}_{attribute_name}_masked_img.jpg'), masked_img)
    cv2.imwrite(os.path.join(output_dir, f'{img_nr}_{attribute_name}_masked_cam.jpg'), masked_img_activation)
    
    return amr, amr_corr, mask_relative, count_activated, count_true_positive, count_false_positive
    
