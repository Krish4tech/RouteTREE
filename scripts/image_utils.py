import cv2
import numpy as np

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class ImageUtils:

    @staticmethod
    def cv_to_pixmap(image, width=500, height=500):
        """
        Convert OpenCV image to QPixmap
        """

        if len(image.shape) == 2:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        h, w, ch = rgb.shape

        bytes_per_line = ch * w

        qt_image = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)

        pixmap = pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        return pixmap

    ############################################################

    @staticmethod
    def save_image(path, image):

        cv2.imwrite(path, image)

    ############################################################

    @staticmethod
    def resize_keep_ratio(image, max_size=512):

        h, w = image.shape[:2]

        scale = min(
            max_size / w,
            max_size / h
        )

        new_w = int(w * scale)

        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h)
        )

        return resized