"""
test_road_gap_detector.py
=========================
Test for RoadGapDetector — Module 1 of the RouteTREE healing pipeline.

HOW TO RUN
----------
    cd D:/Projects/RouteTREE/scripts
    python test_road_gap_detector.py

    OR with a specific mask:
    python test_road_gap_detector.py --mask path/to/your_mask.png

WHAT THIS DOES
--------------
    1. Loads a road mask PNG.
    2. Skeletonizes via GraphUtils.skeletonize().
    3. Detects nodes via NodeDetector.detect_nodes().
    4. Flattens node dict → plain list of (x,y) tuples.
    5. Generates labels via LabelGenerator.generate_labels().
    6. Generates edges via EdgeGenerator.generate_edges().
    7. Builds NetworkX graph via NetworkBuilder.build().
    8. Runs RoadGapDetector at three search radii (30, 50, 75 px).
    9. Prints detailed candidate lists.
   10. Saves three visualization PNGs to outputs/comparisons/.

OUTPUT FILES
------------
    outputs/comparisons/gap_detector_r30.png
    outputs/comparisons/gap_detector_r50.png
    outputs/comparisons/gap_detector_r75.png
"""

import argparse
import os
import sys

import cv2
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — runnable from scripts/ or project root
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import RouteTREE modules
# ---------------------------------------------------------------------------
from graph_utils        import GraphUtils
from node_detector      import NodeDetector
from edge_generator     import EdgeGenerator
from network_builder    import NetworkBuilder
from label_generator    import LabelGenerator
from road_gap_detector  import RoadGapDetector

# ---------------------------------------------------------------------------
# Default test mask
# ---------------------------------------------------------------------------
DEFAULT_MASK = os.path.join(
    PROJECT_ROOT, "datasets", "deepglobe", "masks", "road_mask.png"
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_graph_from_mask(mask: np.ndarray):
    """
    Run the full RouteTREE pipeline on a binary road mask.

    Pipeline
    --------
        GraphUtils.skeletonize(mask)
            ↓
        NodeDetector.detect_nodes(skeleton)        → nodes_dict
            ↓
        detector.get_all_nodes(nodes_dict)         → all_points  (flat list)
            ↓
        LabelGenerator.generate_labels(n)          → labels  ['A','B',...]
            ↓
        EdgeGenerator.generate_edges(skeleton, all_points)   → edges
            ↓
        NetworkBuilder.build(all_points, labels, edges)      → G
            ↓
        positions = { label : (x,y) }

    Returns
    -------
    G         : networkx.Graph      nodes labelled 'A', 'B', ...
    positions : dict                { 'A': (x,y), 'B': (x,y), ... }
    skeleton  : np.ndarray          binary skeleton
    all_points: list                flat list of (x,y) node coords
    nodes_dict: dict                raw output from NodeDetector
    """

    # Step 1 — skeletonize
    print("  [1/5] Skeletonizing mask ...")
    skeleton = GraphUtils.skeletonize(mask)
    print(f"        Skeleton shape    : {skeleton.shape}")
    print(f"        Foreground pixels : {np.count_nonzero(skeleton)}")

    # Step 2 — detect nodes
    print("  [2/5] Detecting nodes ...")
    detector   = NodeDetector()
    nodes_dict = detector.detect_nodes(skeleton)
    counts     = detector.node_count(nodes_dict)
    print(f"        Endpoints : {counts['endpoints']}")
    print(f"        Junctions : {counts['junctions']}")
    print(f"        Turns     : {counts['turns']}")
    print(f"        Total     : {counts['total']}")

    # Step 3 — flatten nodes dict → plain list of (x, y) tuples
    # EdgeGenerator and NetworkBuilder both expect a flat list, not a dict.
    all_points = detector.get_all_nodes(nodes_dict)
    print(f"  [3/5] Flattened to {len(all_points)} node points")

    # Step 4 — generate labels  ('A', 'B', ..., 'Z', 'AA', ...)
    labels = LabelGenerator.generate_labels(len(all_points))
    print(f"  [4/5] Labels: {labels[:5]} ...")

    # Step 5 — generate edges
    print(f"  [5/5] Generating edges ...")
    edges = EdgeGenerator.generate_edges(skeleton, all_points)
    print(f"        Edges found : {len(edges)}")

    # Step 6 — build NetworkX graph
    #   NetworkBuilder.build(points, labels, edges) → nx.Graph
    #   Each node's id is its label string ('A', 'B', ...)
    #   Each node has attribute pos=(x,y)
    G = NetworkBuilder.build(all_points, labels, edges)
    print(f"\n        Graph nodes : {G.number_of_nodes()}")
    print(f"        Graph edges : {G.number_of_edges()}")

    # Build positions dict { label : (x,y) } for visualization
    positions = {label: pt for label, pt in zip(labels, all_points)}

    return G, positions, skeleton, all_points, nodes_dict


# ---------------------------------------------------------------------------
# Detector runner
# ---------------------------------------------------------------------------

def run_detector(
    G          : nx.Graph,
    positions  : dict,
    skeleton   : np.ndarray,
    radius     : int,
    image      : np.ndarray,
) -> list:
    """Run RoadGapDetector at a given radius, print results, save image."""

    print(f"\n{'='*60}")
    print(f"  RoadGapDetector  |  search_radius = {radius} px")
    print(f"{'='*60}")

    detector   = RoadGapDetector(search_radius=radius, min_distance=5.0)
    candidates = detector.find_gaps(G, positions, skeleton)

    # Summary table
    detector.summarize(candidates)

    # Per-candidate detail (cap at 20 lines)
    if candidates:
        show = min(len(candidates), 20)
        print(f"\n  First {show} candidates (of {len(candidates)}):")
        for i, c in enumerate(candidates[:show]):
            print(
                f"    [{i+1:3d}]  "
                f"{str(c['start_node']):>4} {str(c['start_pos']):>14}  →  "
                f"{str(c['end_node']):<4} {str(c['end_pos']):<14}  "
                f"dist = {c['distance']:6.1f} px"
            )

    # Save visualization PNG
    out_path = os.path.join(OUTPUT_DIR, f"gap_detector_r{radius}.png")
    detector.visualize(
        image, positions, candidates,
        G=G, output_path=out_path,
    )

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test RoadGapDetector")
    parser.add_argument(
        "--mask", type=str, default=DEFAULT_MASK,
        help="Path to a binary road mask PNG (white=road, black=background)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load mask
    # ------------------------------------------------------------------
    print(f"\nLoading mask: {args.mask}")
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Cannot load mask: {args.mask}")
        print("        Pass --mask <path> to a valid PNG.")
        sys.exit(1)

    print(f"  Shape      : {mask.shape}")
    print(f"  Road pixels: {np.sum(mask > 127)}")

    # Binarize
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    vis_base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    print(f"\nBuilding road graph ...")
    G, positions, skeleton, all_points, nodes_dict = build_graph_from_mask(mask)

    # ------------------------------------------------------------------
    # Run detector at three radii
    # ------------------------------------------------------------------
    all_candidates = {}
    for radius in [30, 50, 75]:
        all_candidates[radius] = run_detector(
            G, positions, skeleton, radius, vis_base.copy()
        )

    # ------------------------------------------------------------------
    # API sanity checks
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  API sanity checks")
    print(f"{'='*60}")

    detector = RoadGapDetector(search_radius=50)

    # 1. Empty graph returns []
    result = detector.find_gaps(nx.Graph(), {}, skeleton)
    assert result == [], "Empty graph must return []"
    print("  [OK] Empty graph → []")

    # 2. Static helper works
    eps = RoadGapDetector.get_endpoints_from_graph(G)
    assert isinstance(eps, list)
    print(f"  [OK] get_endpoints_from_graph → {len(eps)} endpoints")
    print(f"       Endpoint labels: {eps}")

    # 3. Candidate dict has all required keys
    candidates = detector.find_gaps(G, positions, skeleton)
    required_keys = {
        "start_node", "end_node", "start_pos", "end_pos",
        "distance", "direction_difference", "confidence",
    }
    if candidates:
        missing = required_keys - set(candidates[0].keys())
        assert not missing, f"Missing keys: {missing}"
        print(f"  [OK] All required keys present in candidate dict")

    # 4. direction_difference and confidence are placeholder -1.0
    if candidates:
        assert candidates[0]["direction_difference"] == -1.0
        assert candidates[0]["confidence"] == -1.0
        print("  [OK] Placeholder values correctly set to -1.0")

    # 5. Candidates are sorted by distance
    if len(candidates) > 1:
        dists = [c["distance"] for c in candidates]
        assert dists == sorted(dists), "Candidates must be sorted by distance"
        print("  [OK] Candidates sorted by distance (closest first)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Graph nodes   : {G.number_of_nodes()}")
    print(f"  Graph edges   : {G.number_of_edges()}")
    print(f"  Endpoints     : {len(RoadGapDetector.get_endpoints_from_graph(G))}")
    for r, cands in all_candidates.items():
        print(f"  Gaps (r={r:3d}) : {len(cands)}")
    print(f"\n✓ All checks passed.")
    print(f"  Output images → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
