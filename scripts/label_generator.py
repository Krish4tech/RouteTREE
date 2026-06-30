import cv2
import string


class LabelGenerator:

    """
    Generates labels:

    A
    B
    ...
    Z
    AA
    AB
    ...

    and draws them on an image.
    """

    ##########################################################

    @staticmethod
    def number_to_label(index):

        alphabet = string.ascii_uppercase

        label = ""

        index += 1

        while index > 0:

            index, rem = divmod(index - 1, 26)

            label = alphabet[rem] + label

        return label

    ##########################################################

    @staticmethod
    def generate_labels(node_count):

        labels = []

        for i in range(node_count):

            labels.append(
                LabelGenerator.number_to_label(i)
            )

        return labels

    ##########################################################

    @staticmethod
    def draw_labels(
        image,
        node_points,
        radius=12
    ):

        output = image.copy()

        labels = LabelGenerator.generate_labels(
            len(node_points)
        )

        for (point, label) in zip(node_points, labels):

            x, y = point

            # -------------------------
            # Red Circle
            # -------------------------

            cv2.circle(
                output,
                (x, y),
                radius,
                (0, 0, 255),
                -1
            )

            # White Border

            cv2.circle(
                output,
                (x, y),
                radius,
                (255, 255, 255),
                2
            )

            # -------------------------
            # Font Size
            # -------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX

            scale = 0.45

            thickness = 1

            text_size = cv2.getTextSize(
                label,
                font,
                scale,
                thickness
            )[0]

            text_x = int(
                x - text_size[0] / 2
            )

            text_y = int(
                y + text_size[1] / 2
            )

            # -------------------------
            # White Alphabet
            # -------------------------

            cv2.putText(
                output,
                label,
                (text_x, text_y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return output