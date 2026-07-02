import math
import numpy as np


class Geometry:

    """
    Mathematical utilities for RouteTREE.

    Used by:
        - Node Detector
        - Edge Generator
        - MST
        - DSU
        - Graph Analytics
    """

    ############################################################

    @staticmethod
    def distance(p1, p2):
        """
        Euclidean distance
        """

        x1, y1 = p1
        x2, y2 = p2

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    ############################################################

    @staticmethod
    def midpoint(p1, p2):
        """
        Midpoint between two points
        """

        return (
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2
        )

    ############################################################

    @staticmethod
    def direction_vector(p1, p2):
        """
        Vector from p1 to p2
        """

        return np.array([
            p2[0] - p1[0],
            p2[1] - p1[1]
        ], dtype=np.float32)

    ############################################################

    @staticmethod
    def normalize(vector):
        """
        Unit vector
        """

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector

        return vector / norm

    ############################################################

    @staticmethod
    def angle(p1, p2, p3):
        """
        Angle formed at p2

            p1
             \
              p2 ---- p3
        """

        v1 = Geometry.direction_vector(
            p2,
            p1
        )

        v2 = Geometry.direction_vector(
            p2,
            p3
        )

        v1 = Geometry.normalize(v1)
        v2 = Geometry.normalize(v2)

        dot = np.clip(
            np.dot(v1, v2),
            -1.0,
            1.0
        )

        angle = math.degrees(
            math.acos(dot)
        )

        return angle

    ############################################################

    @staticmethod
    def point_line_distance(point, line_start, line_end):
        """
        Distance of a point from a line
        """

        p = np.array(point, dtype=np.float32)

        a = np.array(line_start, dtype=np.float32)

        b = np.array(line_end, dtype=np.float32)

        if np.array_equal(a, b):
            return Geometry.distance(
                point,
                line_start
            )

        distance = np.abs(
            np.cross(
                b - a,
                a - p
            )
        ) / np.linalg.norm(
            b - a
        )

        return float(distance)

    ############################################################

    @staticmethod
    def is_turn(
        p1,
        p2,
        p3,
        threshold=160
    ):
        """
        Detect whether p2 is a turn.

        Returns:
            True  -> Turn
            False -> Straight
        """

        angle = Geometry.angle(
            p1,
            p2,
            p3
        )

        return angle < threshold

    ############################################################

    @staticmethod
    def centroid(points):
        """
        Center of multiple points
        """

        points = np.array(points)

        x = np.mean(points[:, 0])

        y = np.mean(points[:, 1])

        return (
            int(x),
            int(y)
        )

    ############################################################

    @staticmethod
    def merge_close_points(
        points,
        distance_threshold=8
    ):
        """
        Merge nearby duplicate nodes.
        """

        merged = []

        visited = set()

        for i, p in enumerate(points):

            if i in visited:
                continue

            cluster = [p]

            visited.add(i)

            for j, q in enumerate(points):

                if j in visited:
                    continue

                if Geometry.distance(
                    p,
                    q
                ) < distance_threshold:

                    cluster.append(q)

                    visited.add(j)

            merged.append(
                Geometry.centroid(cluster)
            )

        return merged