# Define your attribute names (in order)
attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
    "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]

# Initialize counters
counts = {attr: {'1': 0, '-1': 0} for attr in attributes}

# Read and parse the file
with open('./CelebA/protocol/list_attr_celeba.txt', 'r') as file:
    lines = file.readlines()

# Skip the header (first two lines)
for line in lines[2:]:
    parts = line.strip().split()
    values = parts[1:]  # skip image filename

    for i, val in enumerate(values):
        if val in ('1', '-1'):
            counts[attributes[i]][val] += 1

# Print results
for attr in attributes:
    print(f"{attr}: 1s = {counts[attr]['1']}, -1s = {counts[attr]['-1']}")
