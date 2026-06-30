import cv2
import numpy as np

from skimage.morphology import skeletonize


class GraphUtils:

    """
    Converts a road mask into a
    one-pixel-wide centerline.
    """

    @staticmethod
    def skeletonize(mask):

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_BGR2GRAY
            )

        binary = (mask > 127).astype(np.uint8)

        skeleton = skeletonize(binary)

        skeleton = (
            skeleton.astype(np.uint8)
            * 255
        )

        return skeleton

    ###########################################################

    @staticmethod
    def remove_small_components(
        skeleton,
        min_area=20
    ):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            skeleton,
            connectivity=8
        )

        output = np.zeros_like(skeleton)

        for i in range(1, num_labels):

            if stats[i, cv2.CC_STAT_AREA] >= min_area:

                output[
                    labels == i
                ] = 255

        return output

    ###########################################################

    @staticmethod
    def dilate_for_display(
        skeleton,
        thickness=2
    ):

        kernel = np.ones(
            (thickness, thickness),
            np.uint8
        )

        return cv2.dilate(
            skeleton,
            kernel,
            iterations=1
        )

    ###########################################################

    @staticmethod
    def generate_graph(mask):

        skeleton = GraphUtils.skeletonize(mask)

        skeleton = GraphUtils.remove_small_components(
            skeleton
        )

        display = GraphUtils.dilate_for_display(
            skeleton
        )

        return {

            "skeleton": skeleton,

            "display": display

        }