import os
import cv2
import torch
import numpy as np

from model import RouteTREEModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Projects\RouteTREE\models\deeplabv3\best_model.pth"

INPUT_IMAGE = r"D:\Projects\RouteTREE\968674_sat.jpg"

OUTPUT_DIR = r"D:\Projects\RouteTREE\outputs\predictions"

IMAGE_SIZE = 512

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("RouteTREE Prediction")
print("=" * 60)
print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = RouteTREEModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.to(device)

model.eval()

print("Model Loaded Successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(INPUT_IMAGE)

original_height, original_width = image.shape[:2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_rgb = cv2.resize(
    image_rgb,
    (IMAGE_SIZE, IMAGE_SIZE)
)

image_rgb = image_rgb.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

image_rgb = (image_rgb - mean) / std

image_tensor = torch.from_numpy(image_rgb)

image_tensor = image_tensor.permute(2, 0, 1)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.float().to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image_tensor)

    prediction = torch.sigmoid(output)

    prediction = (prediction > THRESHOLD).float()

prediction = prediction.squeeze().cpu().numpy()

prediction = prediction.astype(np.uint8) * 255


# ============================================================
# RESIZE TO ORIGINAL SIZE
# ============================================================

prediction = cv2.resize(
    prediction,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST
)


# ============================================================
# SAVE OUTPUT
# ============================================================

image_name = os.path.basename(INPUT_IMAGE)

image_name = os.path.splitext(image_name)[0]

output_path = os.path.join(
    OUTPUT_DIR,
    image_name + "_mask.png"
)

cv2.imwrite(output_path, prediction)

print()

print("=" * 60)

print("Prediction Completed Successfully")

print(f"Mask Saved : {output_path}")

print("=" * 60)