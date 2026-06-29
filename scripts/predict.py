import time
import cv2
import numpy as np
import torch

from model import RouteTREEModel


class RouteTREEPredictor:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = RouteTREEModel()

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)

        self.model.eval()

        print("Model Loaded Successfully")
        print("Device :", self.device)

        if torch.cuda.is_available():
            print("GPU :", torch.cuda.get_device_name(0))

    ####################################################################

    def preprocess(self, image):

        original_h, original_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, (512, 512))

        rgb = rgb.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.float()

        tensor = tensor.to(self.device)

        return tensor, original_h, original_w

    ####################################################################

    def predict(self, image):

        tensor, h, w = self.preprocess(image)

        start = time.time()

        with torch.no_grad():

            output = self.model(tensor)

            output = torch.sigmoid(output)

            output = (output > 0.5).float()

        elapsed = time.time() - start

        mask = output.squeeze().cpu().numpy()

        mask = (mask * 255).astype(np.uint8)

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        road_pixels = np.sum(mask == 255)

        total_pixels = mask.size

        road_percentage = (
            road_pixels / total_pixels
        ) * 100

        return {

            "mask": mask,

            "time": elapsed,

            "road_pixels": int(road_pixels),

            "road_percentage": float(road_percentage)

        }

    ####################################################################

    def overlay(self, image, mask):

        color_mask = np.zeros_like(image)

        color_mask[:, :, 1] = mask

        overlay = cv2.addWeighted(
            image,
            0.7,
            color_mask,
            0.3,
            0
        )

        return overlay