import cv2
import numpy as np


class NodeDetector:

    """
    Detect important graph nodes from
    a one-pixel skeleton.

    Detects:
    - End Points
    - Junctions
    """

    @staticmethod
    def neighbour_count(skeleton, x, y):

        roi = skeleton[
            y - 1:y + 2,
            x - 1:x + 2
        ]

        count = np.count_nonzero(roi)

        # remove center pixel
        count -= 1

        return count

    ##########################################################

    @staticmethod
    def detect_nodes(skeleton):

        endpoints = []

        junctions = []

        h, w = skeleton.shape

        for y in range(1, h - 1):

            for x in range(1, w - 1):

                if skeleton[y, x] == 0:
                    continue

                neighbours = NodeDetector.neighbour_count(
                    skeleton,
                    x,
                    y
                )

                # End point
                if neighbours == 1:

                    endpoints.append((x, y))

                # Junction
                elif neighbours >= 3:

                    junctions.append((x, y))

        return {

            "endpoints": endpoints,

            "junctions": junctions

        }

    ##########################################################

    @staticmethod
    def draw_nodes(image, nodes):

        output = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        # --------------------------

        for x, y in nodes["endpoints"]:

            cv2.circle(
                output,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

        # --------------------------

        for x, y in nodes["junctions"]:

            cv2.circle(
                output,
                (x, y),
                7,
                (255, 0, 0),
                -1
            )

        return output