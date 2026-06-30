import os
import cv2
import matplotlib.pyplot as plt

from graph_utils import GraphUtils

# ==========================================================
# CHANGE THIS TO YOUR MASK LOCATION
# ==========================================================

MASK_PATH = r"D:\Projects\RouteTREE\outputs\predictions\968674_sat_mask.png"

# ==========================================================

if not os.path.exists(MASK_PATH):
    raise FileNotFoundError(
        f"Mask not found:\n{MASK_PATH}"
    )

mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

graph = GraphUtils.generate_graph(mask)

# ==========================================================
# SAVE RESULT
# ==========================================================

OUTPUT = r"D:\Projects\RouteTREE\outputs\skeleton.png"

cv2.imwrite(
    OUTPUT,
    graph["display"]
)

print(f"Skeleton saved at:\n{OUTPUT}")

# ==========================================================
# DISPLAY USING MATPLOTLIB
# ==========================================================

plt.figure(figsize=(8,8))

plt.imshow(
    graph["display"],
    cmap="gray"
)

plt.title("RouteTREE Skeleton")

plt.axis("off")

plt.show()