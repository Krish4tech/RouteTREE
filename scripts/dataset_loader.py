import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class DeepGlobeDataset(Dataset):
    """
    Dataset Loader for DeepGlobe Road Extraction Dataset
    RouteTREE Project
    """

    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([
            file
            for file in os.listdir(image_dir)
            if file.endswith("_sat.jpg")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # -------------------------------------------------
        # File Names
        # -------------------------------------------------

        image_name = self.images[index]
        mask_name = image_name.replace("_sat.jpg", "_mask.png")

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # -------------------------------------------------
        # Read Satellite Image
        # -------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Cannot read image:\n{image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------
        # Read Road Mask
        # -------------------------------------------------

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Cannot read mask:\n{mask_path}")

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        image = cv2.resize(
            image,
            (512, 512),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (512, 512),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------
        # Convert Mask to Binary
        # -------------------------------------------------

        mask = (mask > 127).astype(np.float32)

        # -------------------------------------------------
        # Apply Albumentations
        # -------------------------------------------------

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        else:

            image = image.astype(np.float32) / 255.0

            image = torch.from_numpy(image).permute(2, 0, 1)

            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask