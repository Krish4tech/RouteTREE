import cv2
import matplotlib.pyplot as plt

from graph_utils import GraphUtils
from node_detector import NodeDetector

MASK = r"C:\Users\shrik\Downloads\archive\train\972546_mask.png"

mask = cv2.imread(
    MASK,
    cv2.IMREAD_GRAYSCALE
)

graph = GraphUtils.generate_graph(mask)

nodes = NodeDetector.detect_nodes(
    graph["skeleton"]
)

result = NodeDetector.draw_nodes(
    graph["display"],
    nodes
)

print(
    "Endpoints :",
    len(nodes["endpoints"])
)

print(
    "Junctions :",
    len(nodes["junctions"])
)

plt.figure(figsize=(8,8))

plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.show()