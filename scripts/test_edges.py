import cv2
import matplotlib.pyplot as plt

from graph_utils import GraphUtils
from node_detector import NodeDetector
from edge_generator import EdgeGenerator

MASK = r"C:\Users\shrik\Downloads\archive\train\972546_mask.png"

mask = cv2.imread(
    MASK,
    cv2.IMREAD_GRAYSCALE
)

graph = GraphUtils.generate_graph(mask)

nodes = NodeDetector.detect_nodes(
    graph["skeleton"]
)

points = (
    nodes["endpoints"] +
    nodes["junctions"]
)

edges = EdgeGenerator.generate_edges(
    graph["skeleton"],
    points
)

print()

print("Nodes :", len(points))

print("Edges :", len(edges))

image = cv2.cvtColor(
    graph["display"],
    cv2.COLOR_GRAY2BGR
)

image = EdgeGenerator.draw_edges(
    image,
    edges
)

plt.figure(figsize=(10,10))

plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.show()