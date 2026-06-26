from torch.utils.data import DataLoader

from dataset_loader import DeepGlobeDataset
from augmentation import get_train_augmentation

IMAGE_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\images"
MASK_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\masks"

dataset = DeepGlobeDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=get_train_augmentation()
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

images, masks = next(iter(loader))

print(images.shape)
print(masks.shape)