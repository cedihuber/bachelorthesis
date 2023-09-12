import cv2
import torch
import torch.nn.functional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from importlib.machinery import SourceFileLoader
import torchvision
import os
import numpy as np
from .get_attributes import get_attr
from .ground_truth import get_ground_truth
from tqdm import tqdm


# Taken directly from the thesis of Bieri
class BinaryCategoricalClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return abs(model_output[self.category])
        return abs(model_output[:, self.category])

def load_image(path):
    # load example image (already preprocessed)
    rgb_img = torchvision.io.image.read_image(path)
    # convert to the required data type
    image = rgb_img / 255.
    input_tensor = image.unsqueeze(0)
    rgb_img = image.numpy().transpose(1, 2, 0)

    return input_tensor, rgb_img


def gen_gradcam_all_attributes(img_paths, model_type, nr_imgs, output_dir, bool_cuda, path_gt):
    # load network model
    MainModel = SourceFileLoader("MainModel", "attribute_cam/AFFACT.py").load_module()
    model = torch.load("attribute_cam/AFFACT_" + model_type + ".pth")
    # insert identity layer in order to retrieve the data from res5c
    model.identity = torch.nn.Identity()
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    #target_layers = [model.res5c_branch2c]
    target_layers = [model.identity]
    layer = 'res5c' # also in acceptable_mask_ratio.py
    if bool_cuda:
        model = model.cuda()

    # setup network
    model.eval()

    # dictionary with attribute names
    attribute_names = get_attr()
    numbers = [n for n in range(40)]
    attr_dict = dict(zip(numbers, attribute_names))

    # Create class activation maps
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=bool_cuda) as cam:

        # apply cam algorithm on every image and save it
        for i in tqdm(range(nr_imgs), leave=False):
#                img_number = img_paths[i].split("\\")[-1].split('.')[0]
            img_number = os.path.splitext(os.path.basename(img_paths[i]))[0]
            input_tensor, rgb_img = load_image(img_paths[i])

            ## CAM
            for attribute in range(40):
                output_dir_gradcam = os.path.join(output_dir, model_type, layer, 'gradcam', attribute_names[attribute])

                targets = [BinaryCategoricalClassifierOutputTarget(attribute)]
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                activation = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.0)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cam_bgr = cv2.cvtColor(activation, cv2.COLOR_RGB2BGR)

                # save image

                # check if output folder for the average imgs exists, if not, create it
                if not os.path.isdir(output_dir_gradcam):
                    os.makedirs(output_dir_gradcam, exist_ok=True)

                output_dir_img = os.path.join(output_dir_gradcam, img_number)
                # check if output folder for the gradcam imgs exists, if not, create it
                if not os.path.isdir(output_dir_img):
                    os.makedirs(output_dir_img, exist_ok=True)

                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam_on_img.png'), cam_image_bgr)
                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam.png'), cam_bgr)
                # save grayscale img for calculating acceptable mask ratio
                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_grayscale.png'), grayscale_cam* 255)


def gradcam_average(img_paths, model_type, nr_imgs, output_dir):

    layer = 'res5c'
    # dictionary with attribute names
    attribute_names = get_attr()
    numbers = [n for n in range(40)]
    attr_dict = dict(zip(numbers, attribute_names))


    # apply cam algorithm on every image and save it
    for attribute in tqdm(range(40)):
        output_dir_gradcam = os.path.join(output_dir, model_type, layer, 'gradcam', attribute_names[attribute])
        imgs = np.ndarray((224,224,3))
        activations = np.ndarray((224,224,3))
        gray = np.ndarray((224,224,3))
        for i in range(nr_imgs):
            # load images
            img_number = os.path.splitext(os.path.basename(img_paths[i]))[0]
            output_dir_img = os.path.join(output_dir_gradcam, img_number)
            imgs += cv2.imread(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam_on_img.png'))
            activations += cv2.imread(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam.png'))
            # save grayscale img for calculating acceptable mask ratio
            gray += cv2.imread(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_grayscale.png'))

        # average
        avg_im = np.round(imgs/nr_imgs)
        cv2.imwrite(os.path.join(output_dir_gradcam, f'avg_{attr_dict[attribute]}.png'), avg_im)
        avg_act = np.round(activations/nr_imgs)
        cv2.imwrite(os.path.join(output_dir_gradcam, f'avg_cam_{attr_dict[attribute]}.png'), avg_act)
        avg_gray = np.round(gray/nr_imgs)
        cv2.imwrite(os.path.join(output_dir_gradcam, f'avg_gray_{attr_dict[attribute]}.png'), avg_gray)


def gen_gradcam(img_paths, img_numbers, model_type, attribute, start, end, nr_imgs, output_dir, bool_cuda, path_gt):
    # load network model
    MainModel = SourceFileLoader("MainModel", "attribute_cam/AFFACT.py").load_module()
    model = torch.load("attribute_cam/AFFACT_" + model_type + ".pth")
    # insert identity layer in order to retrieve the data from res5c
    model.identity = torch.nn.Identity()
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    #target_layers = [model.res5c_branch2c]
    target_layers = [model.identity]
#    layer = 'res5c' # also in acceptable_mask_ratio.py

    # setup network
    model.eval()

    # dictionary with attribute names
    attribute_names = get_attr()
    numbers = [n for n in range(40)]
    attr_dict = dict(zip(numbers, attribute_names))

    # define target category (facial attribute)
    #targets = [ClassifierOutputTarget(attribute)]

    # Create class activation maps
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=bool_cuda) as cam:

            ## PREPARATIONS
            # get ground truth values for the current attribute
            ground_truth = get_ground_truth(path_gt)
            s = (start-1) #-1 because index = img_number-1 (img_nr: 162771, idx: 162770)
            e = (end)
            # subset conarining only validation and test partition
            subset = ground_truth[s:e]
            # filter ground truth subset
            imgs_jpg = [str(nr) + '.jpg' for nr in img_numbers]
            subset = subset[subset['image'].isin(imgs_jpg)]
            column = subset[attribute_names[attribute]]
            ground_truth_attribute = column.values.tolist()

            imgs = []
            activations = []
            gray = []

            ## CAM
            # apply cam algorithm on every image and save it
            for i in tqdm(range(nr_imgs), leave=False):
#                img_number = img_paths[i].split("\\")[-1].split('.')[0]
                img_number = os.path.splitext(os.path.basename(img_paths[i]))[0]
                targets = [BinaryCategoricalClassifierOutputTarget(attribute)]
                input_tensor, rgb_img = load_image(img_paths[i])
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                gray.append(grayscale_cam)


                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                activation = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.0)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cam_bgr = cv2.cvtColor(activation, cv2.COLOR_RGB2BGR)
                activations.append(cam_bgr)
                imgs.append(cam_image_bgr)

                # save image

                # check if output folder for the average imgs exists, if not, create it
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                output_dir_img = os.path.join(output_dir, img_number)
                # check if output folder for the gradcam imgs exists, if not, create it
                if not os.path.isdir(output_dir_img):
                    os.makedirs(output_dir_img, exist_ok=True)

                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam_on_img.png'), cam_image_bgr)
                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_cam.png'), cam_bgr)
                # save grayscale img for calculating acceptable mask ratio
                cv2.imwrite(os.path.join(output_dir_img, f'{img_number}_{attr_dict[attribute]}_grayscale.png'), grayscale_cam)


            ## AVERAGE (AND STD)
            # convert to numpy array
            imgs = np.array(imgs, dtype=np.uint8)
            activations = np.array(activations, dtype=np.uint8)
            gray = np.array(gray, dtype=np.uint8)

            # average
            avg_im = np.round(np.mean(imgs, axis=0))
            cv2.imwrite(os.path.join(output_dir, f'avg_{attr_dict[attribute]}.png'), avg_im)
            avg_act = np.round(np.mean(activations, axis=0))
            cv2.imwrite(os.path.join(output_dir, f'avg_cam_{attr_dict[attribute]}.png'), avg_act)
            avg_gray = np.mean(gray, axis=0)
            cv2.imwrite(os.path.join(output_dir, f'avg_gray_{attr_dict[attribute]}.png'), avg_gray)

# =============================================================================
#             # standard deviation
#             std_im = np.std(imgs, axis=0)
#             cv2.imwrite(os.path.join(output_dir, f'std_{start}_{end}_{attr_dict[attribute]}.jpg'), std_im)
#             std_act = np.std(activations, axis=0)
#             cv2.imwrite(os.path.join(output_dir, f'std_cam_{start}_{end}_{attr_dict[attribute]}.jpg'), std_act)
#             if count_t != 0:
#                 std_im_t = np.std(imgs_t, axis=0)
#                 cv2.imwrite(os.path.join(output_dir, f'std_true_{start}_{end}_{attr_dict[attribute]}.jpg'), std_im_t)
#             if count_f != 0:
#                 std_im_f = np.std(imgs_f, axis=0)
#                 cv2.imwrite(os.path.join(output_dir, f'std_false_{start}_{end}_{attr_dict[attribute]}.jpg'), std_im_f)
# =============================================================================
