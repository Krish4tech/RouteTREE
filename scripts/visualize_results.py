import random
import numpy as np
import matplotlib.pyplot as plt

from dataset_loader import DeepGlobeDataset
from augmentation import get_train_augmentation


# ============================================================
# Dataset Paths
# ============================================================

IMAGE_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\images"
MASK_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\masks"


# ============================================================
# Load Datasets
# ============================================================

original_dataset = DeepGlobeDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=None
)

augmented_dataset = DeepGlobeDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=get_train_augmentation()
)


# ============================================================
# Select Random Sample
# ============================================================

index = random.randint(0, len(original_dataset) - 1)

image_name = original_dataset.images[index]

original_image, original_mask = original_dataset[index]
augmented_image, augmented_mask = augmented_dataset[index]
print(type(augmented_image))
print(augmented_image.shape)
print(augmented_image.min())
print(augmented_image.max())


# ============================================================
# Convert Tensors -> NumPy
# ============================================================

original_image = original_image.permute(1, 2, 0).numpy()
original_mask = original_mask.squeeze().numpy()

augmented_image = augmented_image.permute(1, 2, 0).numpy()
augmented_mask = augmented_mask.squeeze().numpy()


# ============================================================
# De-normalize augmented image for display
# ============================================================

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

augmented_image = (augmented_image * std) + mean
augmented_image = np.clip(augmented_image, 0, 1)


# ============================================================
# Calculate Dataset Statistics
# ============================================================

road_pixels = np.sum(original_mask > 0.5)
total_pixels = original_mask.size

road_percentage = (road_pixels / total_pixels) * 100
background_percentage = 100 - road_percentage


# ============================================================
# Display Information
# ============================================================

print("=" * 60)
print("           RouteTREE Dataset Inspector")
print("=" * 60)
print(f"Image Name        : {image_name}")
print(f"Image Resolution  : {original_image.shape[1]} x {original_image.shape[0]}")
print(f"Road Pixels       : {road_percentage:.2f}%")
print(f"Background Pixels : {background_percentage:.2f}%")
print("=" * 60)


# ============================================================
# Plot Images
# ============================================================

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

ax[0, 0].imshow(original_image)
ax[0, 0].set_title("Original Satellite", fontsize=14)
ax[0, 0].axis("off")

ax[0, 1].imshow(original_mask, cmap="gray")
ax[0, 1].set_title("Original Road Mask", fontsize=14)
ax[0, 1].axis("off")

ax[1, 0].imshow(augmented_image)
ax[1, 0].set_title("Augmented Satellite", fontsize=14)
ax[1, 0].axis("off")

ax[1, 1].imshow(augmented_mask, cmap="gray")
ax[1, 1].set_title("Augmented Road Mask", fontsize=14)
ax[1, 1].axis("off")

plt.suptitle(
    f"RouteTREE Dataset Inspector\nSample : {image_name}",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.show()