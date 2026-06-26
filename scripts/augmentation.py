import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation():

    return A.Compose([

        A.Resize(512, 512),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.3),

        A.Rotate(
            limit=30,
            border_mode=0,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        A.GaussianBlur(
            blur_limit=(3,5),
            p=0.3
        ),

        A.GaussNoise(
            std_range=(0.02,0.08),
            p=0.3
        ),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225),
            max_pixel_value=255.0
        ),

        ToTensorV2()

    ])