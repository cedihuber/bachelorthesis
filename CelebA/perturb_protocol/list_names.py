import os

def list_names():
    # Define the folder path
    folder_path = "../../../../local/scratch/chuber/result"
    output_file = "CelebA/perturb_protocol/image_names.txt"
    
    # Get a list of image filenames
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png") and not f.startswith("saliency") and not f.startswith("mask") and os.path.isfile(os.path.join(folder_path, f))]
    
    # Write filenames to a text file
    with open(output_file, "w") as file:
        for image in image_files:
            file.write(image + "\n")
    
    print(f"Image names saved to {output_file}")