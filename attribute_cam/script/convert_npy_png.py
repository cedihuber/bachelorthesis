import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytorch_grad_cam
import torchvision
import torch


def load_img(path, input_size=(224, 224)):
    image = torchvision.io.image.read_image(path)
    # convert to the required data type
    image = image / 255.0
    # add the required batch dimension
    image = image.unsqueeze(0)
    
    return image, image[0].numpy().transpose(1,2,0)

# Load the array
activation = np.load("../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch/Bags_Under_Eyes/182665.png.npy")  # shape (224, 224)

#image, orig_image = load_img("../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch/original_image_182640.png")
orig_image = np.zeros((224, 224, 3), dtype=np.float32)  # all ones = white
print(orig_image.shape)
overlay = pytorch_grad_cam.utils.image.show_cam_on_image(orig_image, activation, use_rgb=True)
torchvision.io.write_png(torch.tensor(overlay.transpose(2,0,1), dtype=torch.uint8), "../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch/Bags_Under_Eyes/182655_activation.png")

# activation = activation - activation.min()
# activation = activation / (activation.max() + 1e-8)
# img_array = (activation * 255).astype(np.uint8)

# plt.imshow(activation, cmap='viridis')  # or 'hot', 'jet', 'plasma', etc.
# plt.axis('off')  # no axes
# plt.tight_layout(pad=0)
# plt.savefig("../../../../local/scratch/chuber/Finalresults/balanced/corrRise_masks_black_3000_masks_30_patch/Bags_Under_Eyes/182640_activation.png", bbox_inches='tight', pad_inches=0)
# plt.close()

# Save as PNG
