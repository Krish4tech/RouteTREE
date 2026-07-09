"""
test_gap_scoring.py
====================

RouteTREE end-to-end test script.

Pipeline under test:
    road_mask.png -> skeleton -> graph
                   -> RoadGapDetector      (Module 1)
                   -> OrientationAnalyzer  (Module 2)
                   -> CandidateMatcher     (Module 3)
                   -> GapScoring           (Module 4)

For each of GapScoring's three combination methods
("weighted_sum", "geometric", "harmonic") this script:
    1. Re-scores an independent copy of the same matched candidate list.
    2. Prints a full per-candidate breakdown table via GapScoring.summarize().
    3. Saves a color-coded confidence visualization PNG to
       outputs/comparisons/.

Run directly:
    python test_gap_scoring.py

NOTE: This file assumes the "already completed" upstream helper modules
(GraphUtils, NodeDetector, LabelGenerator, EdgeGenerator, NetworkBuilder)
live alongside this script using the same one-class-per-file naming
convention as road_gap_detector.py / orientation_analyzer.py /
candidate_matcher.py. If your actual filenames differ, update the
import block below accordingly -- nothing else in this script needs to
change.
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

# --- Module under test ---------------------------------------------------
from gap_scoring import GapScoring


# ----------------------------------------------------------------------
# Configuration -- adjust these paths to match your local project layout
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # D:\Projects\RouteTREE
MASK_PATH = os.path.join(PROJECT_ROOT, "datasets", "deepglobe", "masks", "road_mask.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "comparisons")


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
    the background canvas for GapScoring.visualize() so colored overlay
    lines are visible.
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
    print(f"[test_gap_scoring] RoadGapDetector found {len(candidates)} raw candidate(s).")
    gap_detector.summarize(candidates)

    ep_labels = RoadGapDetector.get_endpoints_from_graph(G)
    ep_coords = [positions[lbl] for lbl in ep_labels if lbl in positions]

    analyzer = OrientationAnalyzer(walk_length=20)
    directions = analyzer.analyze_all(skeleton, ep_coords)
    print(f"[test_gap_scoring] OrientationAnalyzer computed {len(directions)} direction(s).")

    matcher = CandidateMatcher(max_distance=75.0, facing_threshold=120.0)
    matched = matcher.match_candidates(candidates, directions, skeleton)
    accepted = sum(1 for c in matched if not c.get("rejected", False))
    print(
        f"[test_gap_scoring] CandidateMatcher accepted "
        f"{accepted}/{len(matched)} candidate(s)."
    )

    return matched


# ------------------------------------------------------------------
# Module 4: GapScoring, exercised across all 3 combination methods
# ------------------------------------------------------------------

def _independent_copy(candidates):
    """
    Shallow-copy every candidate dict.

    GapScoring.score_candidates() mutates each candidate dict in place
    (adding confidence / confidence_grade / score_breakdown). Since this
    test scores the SAME matched candidate list three times (once per
    combination method), each run needs its own independent copy so
    results from one method don't leak into or get overwritten by the
    next.
    """
    return [dict(c) for c in candidates]


def score_and_visualize(matched_candidates, reference_image, output_dir):
    """
    Runs GapScoring with each of the three combination methods on
    independent copies of matched_candidates, printing a summary table
    and saving a visualization PNG for each.
    """
    combination_methods = ["weighted_sum", "geometric", "harmonic"]

    for method in combination_methods:
        print(f"\n{'=' * 70}")
        print(f"[test_gap_scoring] Scoring with combination method: '{method}'")
        print(f"{'=' * 70}")

        candidates_for_method = _independent_copy(matched_candidates)

        scorer = GapScoring(
            weight_distance=0.35,
            weight_direction=0.30,
            weight_facing=0.35,
            combination=method,
            high_threshold=0.65,
            low_threshold=0.40,
        )
        scored = scorer.score_candidates(candidates_for_method)

        GapScoring.summarize(scored)

        output_path = os.path.join(output_dir, f"gap_scoring_{method}.png")
        GapScoring.visualize(reference_image, scored, output_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[test_gap_scoring] Loading mask from: {MASK_PATH}")
    mask = load_mask(MASK_PATH)
    reference_image = load_reference_image(MASK_PATH)

    print("[test_gap_scoring] Building skeleton + graph...")
    skeleton, G, positions = build_graph_from_mask(mask)
    print(
        f"[test_gap_scoring] Graph built: {G.number_of_nodes()} node(s), "
        f"{G.number_of_edges()} edge(s)."
    )

    print("[test_gap_scoring] Running Modules 1-3 (gap detection / orientation / matching)...")
    matched_candidates = run_gap_pipeline(skeleton, G, positions)

    if not matched_candidates:
        print("[test_gap_scoring] No candidates found -- nothing to score. Exiting.")
        return

    score_and_visualize(matched_candidates, reference_image, OUTPUT_DIR)

    print(f"\n[test_gap_scoring] Done. 3 comparison PNGs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()