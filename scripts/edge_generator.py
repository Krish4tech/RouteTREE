import cv2
import numpy as np


class EdgeGenerator:

    """
    Traverse the road skeleton and
    discover road edges between nodes.
    """

    ##########################################################

    @staticmethod
    def get_neighbors(skeleton, x, y):

        neighbors = []

        h, w = skeleton.shape

        for dy in [-1, 0, 1]:

            for dx in [-1, 0, 1]:

                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if nx < 0 or ny < 0:
                    continue

                if nx >= w or ny >= h:
                    continue

                if skeleton[ny, nx] > 0:
                    neighbors.append((nx, ny))

        return neighbors

    ##########################################################

    @staticmethod
    def build_node_lookup(points):

        lookup = {}

        for point in points:
            lookup[point] = True

        return lookup

    ##########################################################

    @staticmethod
    def trace_edge(
        skeleton,
        start,
        previous,
        node_lookup,
        visited
    ):

        path = [start]

        current = start

        prev = previous

        while True:

            visited.add(current)

            neighbors = EdgeGenerator.get_neighbors(
                skeleton,
                current[0],
                current[1]
            )

            if prev is not None:

                neighbors = [
                    p for p in neighbors
                    if p != prev
                ]

            if len(neighbors) == 0:
                return None

            nxt = neighbors[0]

            path.append(nxt)

            if nxt in node_lookup:

                return {

                    "start": start,

                    "end": nxt,

                    "pixels": path,

                    "length": len(path)

                }

            prev = current

            current = nxt

    ##########################################################

    @staticmethod
    def generate_edges(
        skeleton,
        node_points
    ):

        node_lookup = EdgeGenerator.build_node_lookup(
            node_points
        )

        visited = set()

        edges = []

        for node in node_points:

            neighbors = EdgeGenerator.get_neighbors(
                skeleton,
                node[0],
                node[1]
            )

            for neighbour in neighbors:

                if neighbour in visited:
                    continue

                edge = EdgeGenerator.trace_edge(
                    skeleton,
                    neighbour,
                    node,
                    node_lookup,
                    visited
                )

                if edge is None:
                    continue

                edge["start"] = node

                ##################################################
                # Ignore self-loop
                ##################################################

                if edge["start"] == edge["end"]:
                    continue

                ##################################################
                # Ignore duplicate edges
                ##################################################

                duplicate = False

                for existing in edges:

                    if (
                        existing["start"] == edge["start"]
                        and
                        existing["end"] == edge["end"]
                    ):

                        duplicate = True
                        break

                    if (
                        existing["start"] == edge["end"]
                        and
                        existing["end"] == edge["start"]
                    ):

                        duplicate = True
                        break

                if duplicate:
                    continue

                edges.append(edge)

        return edges

    ##########################################################

    @staticmethod
    def draw_edges(
        image,
        edges,
        color=(0, 255, 0)
    ):

        output = image.copy()

        for edge in edges:

            pts = edge["pixels"]

            for i in range(len(pts) - 1):

                cv2.line(

                    output,

                    pts[i],

                    pts[i + 1],

                    color,

                    2

                )

        return output