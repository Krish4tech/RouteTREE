import cv2

from graph_utils import GraphUtils
from node_detector import NodeDetector
from edge_generator import EdgeGenerator
from label_generator import LabelGenerator
from network_builder import NetworkBuilder

MASK = r"C:\Users\shrik\Downloads\archive\train\972546_mask.png"

mask = cv2.imread(
    MASK,
    cv2.IMREAD_GRAYSCALE
)

graph = GraphUtils.generate_graph(mask)

nodes = NodeDetector.detect_nodes(
    graph["skeleton"]
)

# Get all detected nodes
points = NodeDetector.get_all_nodes(nodes)

# Generate labels
labels = LabelGenerator.generate_labels(
    len(points)
)

# Generate edges
edges = EdgeGenerator.generate_edges(
    graph["skeleton"],
    points
)

# Build graph
G = NetworkBuilder.build(
    points,
    labels,
    edges
)

print("\n==============================")
print("RouteTREE Graph Information")
print("==============================")

print("Endpoints :", len(nodes["endpoints"]))
print("Junctions :", len(nodes["junctions"]))
print("Total Nodes :", NodeDetector.node_count(nodes))
print("Total Edges :", len(edges))

print("\n==============================")
print("Node Labels")
print("==============================")

for label, point in zip(labels, points):
    print(f"{label} -> {point}")

print("\n==============================")
print("Graph Edges")
print("==============================")

for edge in G.edges(data=True):
    print(edge)