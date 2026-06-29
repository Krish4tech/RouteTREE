import sys
import os
import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QMessageBox
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from predict import RouteTREEPredictor
from image_utils import ImageUtils


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = r"D:\Projects\RouteTREE\models\deeplabv3\best_model.pth"


# ============================================================
# GUI
# ============================================================

class RouteTREEGUI(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("RouteTREE - Road Extraction System")

        self.resize(1400, 850)

        self.predictor = None

        self.original_image = None

        self.predicted_mask = None

        self.overlay_image = None

        self.load_model()

        self.build_ui()

    # ========================================================

    def load_model(self):

        try:

            self.predictor = RouteTREEPredictor(MODEL_PATH)

        except Exception as e:

            QMessageBox.critical(
                self,
                "Model Error",
                str(e)
            )

    # ========================================================

    def build_ui(self):

        title = QLabel("RouteTREE")

        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))

        subtitle = QLabel(
            "Occlusion-Aware Road Extraction"
        )

        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle.setFont(QFont("Arial", 11))

        # ----------------------------------------------------

        self.originalLabel = QLabel()

        self.originalLabel.setFixedSize(520,520)

        self.originalLabel.setStyleSheet(
            """
            border:2px solid gray;
            background:white;
            """
        )

        self.originalLabel.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.originalLabel.setText(
            "Original Satellite Image"
        )

        # ----------------------------------------------------

        self.maskLabel = QLabel()

        self.maskLabel.setFixedSize(520,520)

        self.maskLabel.setStyleSheet(
            """
            border:2px solid gray;
            background:white;
            """
        )

        self.maskLabel.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.maskLabel.setText(
            "Predicted Road Mask"
        )

        # ----------------------------------------------------

        imageLayout = QHBoxLayout()

        imageLayout.addWidget(self.originalLabel)

        imageLayout.addWidget(self.maskLabel)

        # ----------------------------------------------------

        self.browseButton = QPushButton(
            "Browse Image"
        )

        self.predictButton = QPushButton(
            "Generate Road Mask"
        )

        self.overlayButton = QPushButton(
            "Overlay Roads"
        )

        self.saveButton = QPushButton(
            "Save Mask"
        )

        self.browseButton.setMinimumHeight(45)
        self.predictButton.setMinimumHeight(45)
        self.overlayButton.setMinimumHeight(45)
        self.saveButton.setMinimumHeight(45)

        buttonLayout = QGridLayout()

        buttonLayout.addWidget(
            self.browseButton,
            0,
            0
        )

        buttonLayout.addWidget(
            self.predictButton,
            0,
            1
        )

        buttonLayout.addWidget(
            self.overlayButton,
            1,
            0
        )

        buttonLayout.addWidget(
            self.saveButton,
            1,
            1
        )

        # ----------------------------------------------------

        self.timeLabel = QLabel(
            "Prediction Time : --"
        )

        self.pixelLabel = QLabel(
            "Road Pixels : --"
        )

        self.percentLabel = QLabel(
            "Road Percentage : --"
        )

        if self.predictor:

            gpu_name = "CPU"

            if self.predictor.device.type == "cuda":

                import torch

                gpu_name = torch.cuda.get_device_name(0)

        else:

            gpu_name = "Model Not Loaded"

        self.gpuLabel = QLabel(
            f"Device : {gpu_name}"
        )

        infoLayout = QVBoxLayout()

        infoLayout.addWidget(self.timeLabel)

        infoLayout.addWidget(self.pixelLabel)

        infoLayout.addWidget(self.percentLabel)

        infoLayout.addWidget(self.gpuLabel)

        # ----------------------------------------------------

        layout = QVBoxLayout()

        layout.addWidget(title)

        layout.addWidget(subtitle)

        layout.addSpacing(10)

        layout.addLayout(imageLayout)

        layout.addSpacing(20)

        layout.addLayout(buttonLayout)

        layout.addSpacing(20)

        layout.addLayout(infoLayout)

        self.setLayout(layout)

            # ========================================================
    # CONNECT BUTTONS
    # ========================================================

    def showEvent(self, event):

        self.browseButton.clicked.connect(self.browse_image)
        self.predictButton.clicked.connect(self.generate_mask)
        self.overlayButton.clicked.connect(self.show_overlay)
        self.saveButton.clicked.connect(self.save_mask)

        super().showEvent(event)

    # ========================================================

    def browse_image(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Satellite Image",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )

        if filename == "":
            return

        self.original_image = cv2.imread(filename)

        if self.original_image is None:

            QMessageBox.warning(
                self,
                "Error",
                "Unable to open image."
            )

            return

        pixmap = ImageUtils.cv_to_pixmap(
            self.original_image,
            520,
            520
        )

        self.originalLabel.setPixmap(pixmap)

        self.maskLabel.clear()

        self.maskLabel.setText(
            "Predicted Road Mask"
        )

        self.predicted_mask = None

        self.overlay_image = None

    # ========================================================

    def generate_mask(self):

        if self.original_image is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Please select a satellite image."
            )

            return

        result = self.predictor.predict(
            self.original_image
        )

        self.predicted_mask = result["mask"]

        self.overlay_image = self.predictor.overlay(
            self.original_image,
            self.predicted_mask
        )

        pixmap = ImageUtils.cv_to_pixmap(
            self.predicted_mask,
            520,
            520
        )

        self.maskLabel.setPixmap(pixmap)

        self.timeLabel.setText(
            f"Prediction Time : {result['time']:.3f} sec"
        )

        self.pixelLabel.setText(
            f"Road Pixels : {result['road_pixels']:,}"
        )

        self.percentLabel.setText(
            f"Road Percentage : {result['road_percentage']:.2f}%"
        )

    # ========================================================

    def show_overlay(self):

        if self.overlay_image is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Generate a road mask first."
            )

            return

        pixmap = ImageUtils.cv_to_pixmap(
            self.overlay_image,
            520,
            520
        )

        self.maskLabel.setPixmap(pixmap)

    # ========================================================

    def save_mask(self):

        if self.predicted_mask is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Nothing to save."
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "road_mask.png",
            "PNG Image (*.png)"
        )

        if filename == "":
            return

        ImageUtils.save_image(
            filename,
            self.predicted_mask
        )

        QMessageBox.information(
            self,
            "Saved",
            "Road mask saved successfully."
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = RouteTREEGUI()

    window.show()

    sys.exit(app.exec())