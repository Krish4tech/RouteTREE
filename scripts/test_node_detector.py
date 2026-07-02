import cv2
from graph_utils import GraphUtils
from node_detector import NodeDetector

# ----------------------------------------------------
MASK = r"C:\Users\shrik\Downloads\archive\train\972546_mask.png"
# ----------------------------------------------------

mask = cv2.imread(MASK, 0)
if mask is None:
    raise Exception("Unable to load mask image.")

# ----------------------------------------------------

graph = GraphUtils.generate_graph(mask)
skeleton = graph["skeleton"]

# ✅ CREATE AN INSTANCE FIRST
detector = NodeDetector()

# ✅ CALL METHODS ON THE INSTANCE
nodes = detector.detect_nodes(skeleton)

# ----------------------------------------------------
print("\n==============================")
print("Node Detector Results")
print("==============================")
print("Endpoints :", len(nodes["endpoints"]))
print("Junctions :", len(nodes["junctions"]))
print("Turns     :", len(nodes["turns"]))
print("Total     :", detector.node_count(nodes))

# ----------------------------------------------------
result = detector.draw_nodes(skeleton, nodes)

cv2.imwrite("node_detector_result.png", result)
print("\nImage saved as node_detector_result.png")