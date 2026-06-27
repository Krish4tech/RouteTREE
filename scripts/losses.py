import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    Recommended for Road Segmentation
    """

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(predictions, targets)

        dice_loss = self.dice(predictions, targets)

        total_loss = bce_loss + dice_loss

        return total_loss