import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

# === CONFIGURATION ===
base_path = '../../../../local/scratch/chuber/Finalresults'

# Folder for each view direction
directions = [
    'balanced/corrRise_masks_black_3000_masks_30_patch/test-pr=1',  # direction[0]
    'balanced/corrRise_masks_black_3000_masks_30_patch/test-pr=-1', # direction[1]
    'balanced/corrRise_masks_black_10_masks_30_patch/test-pr=1',    # direction[2]
    'balanced/corrRise_masks_black_10_masks_30_patch/test-pr=-1'    # direction[3]
]

# All attributes
image_files = [
    'Attractive', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones',
    'Male', 'Heavy_Makeup', 'Wavy_Hair', 'Oval_Face', 'Pointy_Nose',
    'Arched_Eyebrows', 'Big_Lips', 'Black_Hair', 'Big_Nose', 'Young',
    'Straight_Hair', 'Brown_Hair', 'Bags_Under_Eyes', 'Wearing_Earrings', 'No_Beard',
    'Bangs', 'Blond_Hair', 'Bushy_Eyebrows', 'Wearing_Necklace', 'Narrow_Eyes',
    '5_o_Clock_Shadow', 'Receding_Hairline', 'Wearing_Necktie', 'Rosy_Cheeks', 'Eyeglasses',
    'Goatee', 'Chubby', 'Sideburns', 'Blurry', 'Wearing_Hat',
    'Double_Chin', 'Pale_Skin', 'Gray_Hair', 'Mustache', 'Bald'
]

# === GRID SIZE ===
num_columns = 10
num_rows = 16

# === CREATE FIGURE ===
fig = plt.figure(figsize=(num_columns * 0.55, num_rows * 0.55))
gs = gridspec.GridSpec(
    num_rows, num_columns,
    figure=fig,
    wspace=0.01,
    hspace=0.05
)

# === PLOT IMAGES ===
image_idx = 0
for row in range(num_rows):
    # Pick the direction pair for this row
    dir_pair = directions[0:2] if row % 2 == 0 else directions[2:4]
    
    for col in range(num_columns):
        if image_idx >= len(image_files):
            break  # Stop if we run out of images
        
        direction = dir_pair[col % 2]
        img_name = image_files[image_idx]
        img_path = os.path.join(base_path, direction, f"{img_name}.png")
        
        # Load and plot image
        ax = fig.add_subplot(gs[row, col])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        image_idx += 1

# === SAVE ===
plt.tight_layout()
plt.savefig('../../../../local/scratch/chuber/Finalresults/grid_10x16.png', dpi=300)
plt.show()
