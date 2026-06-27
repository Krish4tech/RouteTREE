import os
import torch
from tqdm import tqdm

from metrics import SegmentationMetrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        scheduler,
        device,
        model_dir,
    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.scheduler = scheduler

        self.device = device

        self.metrics = SegmentationMetrics()

        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)

        self.best_iou = 0

    #######################################################

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        )

        for images, masks in progress:

            images = images.to(self.device)

            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, masks)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss

    #######################################################

    def validate(self):

        self.model.eval()

        running_loss = 0

        dice_total = 0

        iou_total = 0

        accuracy_total = 0

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False
            )

            for images, masks in progress:

                images = images.to(self.device)

                masks = masks.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)

                running_loss += loss.item()

                dice_total += self.metrics.dice_score(
                    outputs,
                    masks
                )

                iou_total += self.metrics.iou_score(
                    outputs,
                    masks
                )

                accuracy_total += self.metrics.pixel_accuracy(
                    outputs,
                    masks
                )

        val_loss = running_loss / len(self.val_loader)

        dice = dice_total / len(self.val_loader)

        iou = iou_total / len(self.val_loader)

        accuracy = accuracy_total / len(self.val_loader)

        return (
            val_loss,
            dice,
            iou,
            accuracy,
        )

    #######################################################

    def fit(self, epochs):

        for epoch in range(epochs):

            print("\n" + "=" * 60)

            print(f"Epoch {epoch+1}/{epochs}")

            print("=" * 60)

            train_loss = self.train_one_epoch()

            val_loss, dice, iou, accuracy = self.validate()

            if self.scheduler is not None:

                self.scheduler.step(val_loss)

            print(f"Train Loss      : {train_loss:.5f}")

            print(f"Validation Loss : {val_loss:.5f}")

            print(f"Dice Score      : {dice:.4f}")

            print(f"IoU Score       : {iou:.4f}")

            print(f"Pixel Accuracy  : {accuracy:.4f}")

            if iou > self.best_iou:

                self.best_iou = iou

                torch.save(

                    self.model.state_dict(),

                    os.path.join(

                        self.model_dir,

                        "best_model.pth"

                    )

                )

                print("\nBest Model Saved.")

        torch.save(

            self.model.state_dict(),

            os.path.join(

                self.model_dir,

                "last_model.pth"

            )

        )

        print("\nTraining Finished Successfully.")