import os
import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset_loader import DeepGlobeDataset
from augmentation import get_train_augmentation
from model import RouteTREEModel
from losses import BCEDiceLoss
from trainer import Trainer


# ============================================================
# GPU Configuration
# ============================================================

torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RouteTREE Training")
print("=" * 60)
print(f"Device : {device}")

if torch.cuda.is_available():

    print(f"GPU : {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ============================================================
# Dataset Paths
# ============================================================

IMAGE_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\images"

MASK_DIR = r"D:\Projects\RouteTREE\datasets\deepglobe\masks"

MODEL_DIR = r"D:\Projects\RouteTREE\models\deeplabv3"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Hyperparameters
# ============================================================

IMAGE_SIZE = 512

BATCH_SIZE = 4

LEARNING_RATE = 1e-4

EPOCHS = 30

TRAIN_SPLIT = 0.80

VALID_SPLIT = 0.20


# ============================================================
# Dataset
# ============================================================

dataset = DeepGlobeDataset(

    image_dir=IMAGE_DIR,

    mask_dir=MASK_DIR,

    transform=get_train_augmentation()

)

print(f"Total Images : {len(dataset)}")


# ============================================================
# Train / Validation Split
# ============================================================

train_size = int(TRAIN_SPLIT * len(dataset))

valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(

    dataset,

    [train_size, valid_size]

)

print(f"Training Images   : {len(train_dataset)}")

print(f"Validation Images : {len(valid_dataset)}")


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True

)

valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)


# ============================================================
# Model
# ============================================================

model = RouteTREEModel()

model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = BCEDiceLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-5

)


# ============================================================
# Scheduler
# ============================================================

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3

)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    train_loader=train_loader,

    val_loader=valid_loader,

    scheduler=scheduler,

    device=device,

    model_dir=MODEL_DIR

)


# ============================================================
# Start Training
# ============================================================

trainer.fit(EPOCHS)

print("\nTraining Completed Successfully.")

print("Best Model Saved Inside:")

print(MODEL_DIR)

#12 hrs training 30 epoches!!