import os
import shutil

# ==============================
# SOURCE (Your downloaded dataset)
# ==============================
source_folder = r"C:\Users\shrik\Downloads\archive\train"

# ==============================
# DESTINATION (Your project)
# ==============================
image_dest = r"D:\Projects\RouteTREE\datasets\deepglobe\images"
mask_dest = r"D:\Projects\RouteTREE\datasets\deepglobe\masks"

# Create destination folders if they don't exist
os.makedirs(image_dest, exist_ok=True)
os.makedirs(mask_dest, exist_ok=True)

image_count = 0
mask_count = 0

print("Copying files...\n")

for file in os.listdir(source_folder):

    source_path = os.path.join(source_folder, file)

    # Copy satellite images
    if file.endswith("_sat.jpg"):
        shutil.copy2(source_path, os.path.join(image_dest, file))
        image_count += 1

    # Copy masks
    elif file.endswith("_mask.png"):
        shutil.copy2(source_path, os.path.join(mask_dest, file))
        mask_count += 1

print("\nDone!")
print(f"Images copied : {image_count}")
print(f"Masks copied  : {mask_count}")