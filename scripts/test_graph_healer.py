"""
test_graph_healer.py
=====================

RouteTREE end-to-end test script.

Pipeline under test:
    road_mask.png -> skeleton -> graph
                   -> RoadGapDetector      (Module 1)
                   -> OrientationAnalyzer  (Module 2)
                   -> CandidateMatcher     (Module 3)
                   -> GapScoring           (Module 4)
                   -> GraphHealer          (Module 5)

This script:
    1. Rebuilds the skeleton/graph from road_mask.png.
    2. Runs Modules 1-3 to get a matched candidate list.
    3. Scores that list with GapScoring (weighted_sum combination).
    4. Prints the graph's edge count BEFORE healing.
    5. Runs GraphHealer.heal_graph() to insert new edges for
       high-confidence candidates.
    6. Prints the graph's edge count AFTER healing, plus the
       healed/skipped summary dict.
    7. Saves a color-coded visualization to
       outputs/comparisons/graph_healer_result.png.

Run directly:
    python test_graph_healer.py
"""

import os
import sys
import cv2

# ----------------------------------------------------------------------
# Make sure this script's own directory is importable, regardless of the
# working directory it's launched from.
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# --- Upstream pipeline modules (already completed elsewhere) -----------
from graph_utils import GraphUtils
from node_detector import NodeDetector
from label_generator import LabelGenerator
from edge_generator import EdgeGenerator
from network_builder import NetworkBuilder
from road_gap_detector import RoadGapDetector
from orientation_analyzer import OrientationAnalyzer
from candidate_matcher import CandidateMatcher
from gap_scoring import GapScoring

# --- Module under test ---------------------------------------------------
from graph_healer import GraphHealer


# ----------------------------------------------------------------------
# Configuration -- matches the layout confirmed in test_gap_scoring.py
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # D:\Projects\RouteTREE
MASK_PATH = os.path.join(PROJECT_ROOT, "datasets", "deepglobe", "masks", "road_mask.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "comparisons")
RESULT_PATH = os.path.join(OUTPUT_DIR, "graph_healer_result.png")

# Confidence threshold passed to GraphHealer -- only candidates scoring
# at or above this value (and not upstream-rejected) get healed.
HEAL_CONFIDENCE_THRESHOLD = 0.50


# ------------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------------

def load_mask(mask_path):
    """Load the binary road mask from disk as a single-channel image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(
            f"Could not load road mask at '{mask_path}'. "
            f"Update MASK_PATH at the top of this script to point at "
            f"your actual road_mask.png."
        )
    return mask


def load_reference_image(mask_path):
    """
    Load a 3-channel BGR version of the same mask file, used purely as
    the background canvas for GraphHealer.visualize_healed_edges() so
    colored overlay lines are visible.
    """
    image = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load reference image at '{mask_path}'.")
    return image


# ------------------------------------------------------------------
# Graph construction (pre-Module-1 setup, per the confirmed pipeline)
# ------------------------------------------------------------------

def build_graph_from_mask(mask):
    """
    skeletonize -> detect nodes -> generate labels -> generate edges
    -> build graph -> build positions dict.

    Mirrors the confirmed working pipeline call order exactly.
    """
    skeleton = GraphUtils.skeletonize(mask)

    detector = NodeDetector()
    nodes_dict = detector.detect_nodes(skeleton)          # endpoints/junctions/turns
    all_points = detector.get_all_nodes(nodes_dict)        # flat list of (x, y)

    labels = LabelGenerator.generate_labels(len(all_points))     # ["A", "B", ...]
    edges = EdgeGenerator.generate_edges(skeleton, all_points)
    G = NetworkBuilder.build(all_points, labels, edges)           # nx.Graph

    positions = {label: pt for label, pt in zip(labels, all_points)}

    return skeleton, G, positions


# ------------------------------------------------------------------
# Modules 1-3: gap detection -> orientation analysis -> matching
# ------------------------------------------------------------------

def run_gap_pipeline(skeleton, G, positions):
    """
    Runs RoadGapDetector -> OrientationAnalyzer -> CandidateMatcher
    exactly as specified in the confirmed pipeline call order, and
    returns the final enriched/matched candidate list.
    """
    gap_detector = RoadGapDetector(search_radius=75)
    candidates = gap_detector.find_gaps(G, positions, skeleton)
    print(f"[test_graph_healer] RoadGapDetector found {len(candidates)} raw candidate(s).")

    ep_labels = RoadGapDetector.get_endpoints_from_graph(G)
    ep_coords = [positions[lbl] for lbl in ep_labels if lbl in positions]

    analyzer = OrientationAnalyzer(walk_length=20)
    directions = analyzer.analyze_all(skeleton, ep_coords)
    print(f"[test_graph_healer] OrientationAnalyzer computed {len(directions)} direction(s).")

    matcher = CandidateMatcher(max_distance=75.0, facing_threshold=120.0)
    matched = matcher.match_candidates(candidates, directions, skeleton)
    accepted = sum(1 for c in matched if not c.get("rejected", False))
    print(
        f"[test_graph_healer] CandidateMatcher accepted "
        f"{accepted}/{len(matched)} candidate(s)."
    )

    return matched


# ------------------------------------------------------------------
# Module 4: GapScoring
# ------------------------------------------------------------------

def score_candidates(matched_candidates):
    """
    Scores the matched candidate list with GapScoring's default
    "weighted_sum" combination method, printing a summary table before
    handing the scored list off to GraphHealer.
    """
    scorer = GapScoring(
        weight_distance=0.35,
        weight_direction=0.30,
        weight_facing=0.35,
        combination="weighted_sum",
        high_threshold=0.65,
        low_threshold=0.40,
    )
    scored = scorer.score_candidates(matched_candidates)

    print("\n[test_graph_healer] GapScoring results (feeding into GraphHealer):")
    GapScoring.summarize(scored)

    return scored


# ------------------------------------------------------------------
# Module 5: GraphHealer
# ------------------------------------------------------------------

def heal_and_report(G, scored_candidates, positions, reference_image):
    """
    Runs GraphHealer.heal_graph(), prints before/after edge counts and
    the healed/skipped summary, then saves the visualization PNG.

    `positions` must already map every node label referenced by G or by
    scored_candidates to an (x, y) pixel coordinate -- main() is
    responsible for topping it up with any candidate endpoints that
    weren't already part of the original graph before calling this.
    """
    edges_before = G.number_of_edges()
    nodes_before = G.number_of_nodes()

    healer = GraphHealer(confidence_threshold=HEAL_CONFIDENCE_THRESHOLD)
    G, summary = healer.heal_graph(G, scored_candidates)

    edges_after = G.number_of_edges()
    nodes_after = G.number_of_nodes()

    print(f"\n{'=' * 70}")
    print("[test_graph_healer] GraphHealer results")
    print(f"{'=' * 70}")
    print(f"Confidence threshold used : {HEAL_CONFIDENCE_THRESHOLD:.2f}")
    print(f"Nodes  before healing     : {nodes_before}")
    print(f"Nodes  after healing      : {nodes_after}")
    print(f"Edges  before healing     : {edges_before}")
    print(f"Edges  after healing      : {edges_after}")
    print(f"Edges  actually inserted  : {edges_after - edges_before}")
    print("-" * 70)
    print(f"Healed                    : {summary['healed']}")
    print(f"Skipped (low confidence)  : {summary['skipped_low_confidence']}")
    print(f"Skipped (already existed) : {summary['skipped_existing']}")
    print(f"{'=' * 70}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    healer.visualize_healed_edges(
        reference_image, G, positions, scored_candidates, RESULT_PATH
    )

    return G, summary


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[test_graph_healer] Loading mask from: {MASK_PATH}")
    mask = load_mask(MASK_PATH)
    reference_image = load_reference_image(MASK_PATH)

    print("[test_graph_healer] Building skeleton + graph...")
    skeleton, G, positions = build_graph_from_mask(mask)
    print(
        f"[test_graph_healer] Graph built: {G.number_of_nodes()} node(s), "
        f"{G.number_of_edges()} edge(s)."
    )

    print("[test_graph_healer] Running Modules 1-3 (gap detection / orientation / matching)...")
    matched_candidates = run_gap_pipeline(skeleton, G, positions)

    if not matched_candidates:
        print("[test_graph_healer] No candidates found -- nothing to heal. Exiting.")
        return

    print("[test_graph_healer] Running Module 4 (GapScoring)...")
    scored_candidates = score_candidates(matched_candidates)

    # Make sure every candidate endpoint has a known pixel position so
    # the final visualization can actually draw it, even for nodes that
    # were created fresh during healing.
    for candidate in scored_candidates:
        start_node = candidate.get("start_node")
        end_node = candidate.get("end_node")
        if start_node is not None and start_node not in positions and candidate.get("start_pos") is not None:
            positions[start_node] = candidate["start_pos"]
        if end_node is not None and end_node not in positions and candidate.get("end_pos") is not None:
            positions[end_node] = candidate["end_pos"]

    print("[test_graph_healer] Running Module 5 (GraphHealer)...")
    G, summary = heal_and_report(G, scored_candidates, positions, reference_image)

    print(f"[test_graph_healer] Done. Visualization saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main() #testincrement