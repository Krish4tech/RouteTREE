import cv2
import matplotlib.pyplot as plt

from graph_utils import GraphUtils
from node_detector import NodeDetector
from label_generator import LabelGenerator

MASK = r"C:\Users\shrik\Downloads\archive\train\972254_mask.png"

mask = cv2.imread(
    MASK,
    cv2.IMREAD_GRAYSCALE
)

graph = GraphUtils.generate_graph(mask)

nodes = NodeDetector.detect_nodes(
    graph["skeleton"]
)

# Combine endpoints and junctions
points = (
    nodes["endpoints"] +
    nodes["junctions"]
)

image = cv2.cvtColor(
    graph["display"],
    cv2.COLOR_GRAY2BGR
)

image = LabelGenerator.draw_labels(
    image,
    points
)

plt.figure(figsize=(8,8))

plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.show()