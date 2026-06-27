import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class RouteTREEModel(nn.Module):
    """
    RouteTREE Baseline Model

    DeepLabV3 + ResNet50 Backbone

    Later this class will be upgraded with:
        • Transformer Encoder
        • Attention Modules
        • Connectivity Loss
        • Graph-aware Learning
    """

    def __init__(self):

        super().__init__()

        # Load pretrained DeepLabV3
        self.model = deeplabv3_resnet50(
            weights="DEFAULT"
        )

        # Replace classifier for binary segmentation
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):

        output = self.model(x)

        return output["out"]