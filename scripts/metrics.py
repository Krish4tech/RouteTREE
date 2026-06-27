import torch


class SegmentationMetrics:
    """
    Metrics for Binary Road Segmentation
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def dice_score(self, predictions, targets):

        predictions = torch.sigmoid(predictions)
        predictions = (predictions > self.threshold).float()

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            predictions.sum() +
            targets.sum() +
            1e-6
        )

        return dice.item()

    def iou_score(self, predictions, targets):

        predictions = torch.sigmoid(predictions)
        predictions = (predictions > self.threshold).float()

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        union = (
            predictions.sum() +
            targets.sum() -
            intersection
        )

        iou = (
            intersection + 1e-6
        ) / (
            union + 1e-6
        )

        return iou.item()

    def pixel_accuracy(self, predictions, targets):

        predictions = torch.sigmoid(predictions)
        predictions = (predictions > self.threshold).float()

        correct = (predictions == targets).sum().float()

        total = targets.numel()

        accuracy = correct / total

        return accuracy.item()