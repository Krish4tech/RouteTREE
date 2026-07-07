"""
test_candidate_matcher.py
==========================
Test for CandidateMatcher — Module 3 of the RouteTREE healing pipeline.

HOW TO RUN
----------
    cd D:/Projects/RouteTREE/scripts
    python test_candidate_matcher.py

    OR:
    python test_candidate_matcher.py --mask path/to/mask.png

WHAT THIS DOES
--------------
    1. Loads a road mask and builds the full pipeline.
    2. Detects gap candidates (Module 1).
    3. Computes endpoint directions (Module 2).
    4. Runs CandidateMatcher on all candidates (Module 3).
    5. Prints per-candidate enriched results.
    6. Saves a comparison visualization.
    7. Runs synthetic unit tests for each scoring function.

OUTPUT FILES
------------
    outputs/comparisons/matcher_result.png
"""

import argparse
import os
import sys
import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from graph_utils          import GraphUtils
from node_detector        import NodeDetector
from edge_generator       import EdgeGenerator
from network_builder      import NetworkBuilder
from label_generator      import LabelGenerator
from road_gap_detector    import RoadGapDetector
from orientation_analyzer import OrientationAnalyzer
from candidate_matcher    import CandidateMatcher

DEFAULT_MASK = os.path.join(
    PROJECT_ROOT, "datasets", "deepglobe", "masks", "road_mask.png"
)

PASS = "✓"
FAIL = "✗"


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def build_pipeline(mask: np.ndarray):
    """Full pipeline → (G, positions, skeleton, all_points, nodes_dict)."""
    skeleton   = GraphUtils.skeletonize(mask)
    detector   = NodeDetector()
    nodes_dict = detector.detect_nodes(skeleton)
    all_points = detector.get_all_nodes(nodes_dict)
    labels     = LabelGenerator.generate_labels(len(all_points))
    edges      = EdgeGenerator.generate_edges(skeleton, all_points)
    G          = NetworkBuilder.build(all_points, labels, edges)
    positions  = {label: pt for label, pt in zip(labels, all_points)}
    return G, positions, skeleton, all_points, nodes_dict


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------

def test_synthetic():
    print(f"\n{'='*60}")
    print("  Synthetic unit tests for CandidateMatcher")
    print(f"{'='*60}")

    matcher = CandidateMatcher(
        max_distance        = 75.0,
        facing_threshold    = 120.0,
        direction_threshold = 60.0,
    )
    results = []

    # ------------------------------------------------------------------
    # Test 1: compare_distance — close gap scores high
    # ------------------------------------------------------------------
    s1a = matcher.compare_distance(5.0)    # very close → high score
    s1b = matcher.compare_distance(75.0)   # at max     → 0
    s1c = matcher.compare_distance(37.5)   # midpoint   → medium
    ok1 = s1a > 0.8 and s1b == 0.0 and 0.1 < s1c < 0.9
    print(f"\n  Test 1 – compare_distance()")
    print(f"    dist=5.0   → {s1a:.3f}  (expect >0.8)")
    print(f"    dist=75.0  → {s1b:.3f}  (expect 0.0)")
    print(f"    dist=37.5  → {s1c:.3f}  (expect 0.1–0.9)")
    print(f"    Result : {PASS if ok1 else FAIL}")
    results.append(ok1)

    # ------------------------------------------------------------------
    # Test 2: compare_direction — same axis scores 1.0
    # ------------------------------------------------------------------
    s2a = CandidateMatcher.compare_direction(0.0)    # same axis  → 1.0
    s2b = CandidateMatcher.compare_direction(90.0)   # perp       → 0.0
    s2c = CandidateMatcher.compare_direction(45.0)   # 45° off    → ~0.7
    ok2 = (abs(s2a - 1.0) < 0.01 and
           abs(s2b - 0.0) < 0.01 and
           0.5 < s2c < 0.9)
    print(f"\n  Test 2 – compare_direction()")
    print(f"    axis_diff=0°   → {s2a:.3f}  (expect 1.0)")
    print(f"    axis_diff=90°  → {s2b:.3f}  (expect 0.0)")
    print(f"    axis_diff=45°  → {s2c:.3f}  (expect 0.5–0.9)")
    print(f"    Result : {PASS if ok2 else FAIL}")
    results.append(ok2)

    # ------------------------------------------------------------------
    # Test 3: compare_angle — perfectly facing scores 1.0
    # ------------------------------------------------------------------
    s3a = CandidateMatcher.compare_angle(180.0)   # perfect facing → 1.0
    s3b = CandidateMatcher.compare_angle(90.0)    # perpendicular  → 0.0
    s3c = CandidateMatcher.compare_angle(45.0)    # same direction → 0.0
    s3d = CandidateMatcher.compare_angle(135.0)   # partial facing → ~0.5
    ok3 = (abs(s3a - 1.0) < 0.01 and
           abs(s3b - 0.0) < 0.01 and
           s3c == 0.0 and
           0.3 < s3d < 0.7)
    print(f"\n  Test 3 – compare_angle()")
    print(f"    facing=180° → {s3a:.3f}  (expect 1.0)")
    print(f"    facing= 90° → {s3b:.3f}  (expect 0.0)")
    print(f"    facing= 45° → {s3c:.3f}  (expect 0.0)")
    print(f"    facing=135° → {s3d:.3f}  (expect ~0.5)")
    print(f"    Result : {PASS if ok3 else FAIL}")
    results.append(ok3)

    # ------------------------------------------------------------------
    # Test 4: Full match on a perfect synthetic gap
    # Two road segments interrupted in the middle, facing each other.
    # ------------------------------------------------------------------
    skel4 = np.zeros((100, 200), dtype=np.uint8)
    skel4[50, 10:80]  = 1   # left segment  endpoint at (10,50)
    skel4[50, 120:190] = 1   # right segment endpoint at (190,50)

    analyzer  = OrientationAnalyzer(walk_length=20)
    dir_left  = analyzer.estimate_direction(skel4, (10, 50))
    dir_right = analyzer.estimate_direction(skel4, (190, 50))
    directions = {(10, 50): dir_left, (190, 50): dir_right}

    # Build a fake candidate between the two endpoints
    # (distance = 110px — beyond default 75, but we test at max=150)
    matcher4 = CandidateMatcher(max_distance=200.0, facing_threshold=120.0)
    candidate = {
        "start_node"           : "A",
        "end_node"             : "B",
        "start_pos"            : (10, 50),
        "end_pos"              : (190, 50),
        "distance"             : 180.0,
        "direction_difference" : -1.0,
        "confidence"           : -1.0,
    }
    matched = matcher4.match_candidates([candidate], directions, skel4)
    c = matched[0]
    facing  = c.get("facing_angle", 0)
    ok4 = (
        not c["rejected"] and
        abs(facing - 180.0) <= 10.0 and
        c["similarity_score"] > 0.0
    )
    print(f"\n  Test 4 – Perfect synthetic gap (facing endpoints)")
    print(f"    rejected        : {c['rejected']}       (expect False)")
    print(f"    facing_angle    : {facing:.1f}°          (expect ~180°)")
    print(f"    similarity      : {c['similarity_score']:.3f}")
    print(f"    direction_diff  : {c['direction_difference']:.1f}°")
    print(f"    Result : {PASS if ok4 else FAIL}")
    results.append(ok4)

    # ------------------------------------------------------------------
    # Test 5: Reject — same-direction endpoints (not a gap)
    # Both endpoints point East → facing_angle ≈ 0° → rejected
    # ------------------------------------------------------------------
    skel5 = np.zeros((100, 200), dtype=np.uint8)
    skel5[30, 10:80]  = 1   # segment A pointing East
    skel5[70, 10:80]  = 1   # segment B also pointing East (parallel)

    dir_a5 = analyzer.estimate_direction(skel5, (79, 30))  # right end of seg A
    dir_b5 = analyzer.estimate_direction(skel5, (79, 70))  # right end of seg B
    dirs5  = {(79, 30): dir_a5, (79, 70): dir_b5}

    cand5 = {
        "start_node": "A", "end_node": "B",
        "start_pos": (79, 30), "end_pos": (79, 70),
        "distance": 40.0,
        "direction_difference": -1.0, "confidence": -1.0,
    }
    matched5 = matcher.match_candidates([cand5], dirs5, skel5)
    ok5 = matched5[0]["rejected"]
    print(f"\n  Test 5 – Same-direction endpoints (should be REJECTED)")
    print(f"    rejected      : {matched5[0]['rejected']}    (expect True)")
    print(f"    facing_angle  : {matched5[0].get('facing_angle', 'N/A')}°")
    print(f"    reject_reason : {matched5[0].get('reject_reason', '')[:60]}")
    print(f"    Result : {PASS if ok5 else FAIL}")
    results.append(ok5)

    print(f"\n  Passed {sum(results)}/{len(results)} synthetic tests")
    return all(results)


# ---------------------------------------------------------------------------
# Real mask test
# ---------------------------------------------------------------------------

def test_real_mask(mask_path: str):
    print(f"\n{'='*60}")
    print("  Real mask test")
    print(f"{'='*60}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Cannot load: {mask_path}")
        return
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    vis_base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    print("Building pipeline ...")
    G, positions, skeleton, all_points, nodes_dict = build_pipeline(mask)

    # Module 1: Gap detection
    gap_detector = RoadGapDetector(search_radius=75, min_distance=5.0)
    candidates   = gap_detector.find_gaps(G, positions, skeleton)
    print(f"  Gap candidates   : {len(candidates)}")

    # Module 2: Direction analysis (on all endpoint coords)
    ep_labels = RoadGapDetector.get_endpoints_from_graph(G)
    ep_coords = [positions[lbl] for lbl in ep_labels if lbl in positions]
    analyzer  = OrientationAnalyzer(walk_length=20)
    directions = analyzer.analyze_all(skeleton, ep_coords)
    print(f"  Directions computed : {len(directions)}")

    # Module 3: Candidate matching
    matcher = CandidateMatcher(
        max_distance        = 75.0,
        facing_threshold    = 120.0,
        direction_threshold = 60.0,
    )
    matched = matcher.match_candidates(candidates, directions, skeleton)

    accepted = [c for c in matched if not c["rejected"]]
    rejected = [c for c in matched if c["rejected"]]

    print(f"\n  Results:")
    print(f"    Total candidates : {len(matched)}")
    print(f"    Accepted         : {len(accepted)}")
    print(f"    Rejected         : {len(rejected)}")

    # Print accepted candidates in detail
    if accepted:
        print(f"\n  Accepted candidates:")
        hdr = f"  {'Pair':>20}  {'Dist':>6}  {'Facing':>7}  {'AxisDiff':>9}  {'Score':>6}"
        print(hdr)
        print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*6}")
        for c in accepted:
            pair = f"{c['start_node']}→{c['end_node']}"
            print(
                f"  {pair:>20}  "
                f"{c['distance']:>6.1f}px  "
                f"{c['facing_angle']:>6.1f}°  "
                f"{c['direction_difference']:>8.1f}°  "
                f"{c['similarity_score']:>6.3f}"
            )

    # Print rejected candidates
    if rejected:
        print(f"\n  Rejected candidates:")
        for c in rejected:
            pair = f"{c['start_node']}→{c['end_node']}"
            print(f"    {pair:>12}  dist={c['distance']:.1f}px  "
                  f"reason: {c.get('reject_reason','')[:55]}")

    # Visualization
    out_path = os.path.join(OUTPUT_DIR, "matcher_result.png")
    matcher.visualize(vis_base.copy(), matched, output_path=out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test CandidateMatcher")
    parser.add_argument("--mask", type=str, default=DEFAULT_MASK)
    args = parser.parse_args()

    synthetic_ok = test_synthetic()
    test_real_mask(args.mask)

    print(f"\n{'='*60}")
    print(f"  Synthetic tests : {'ALL PASSED' if synthetic_ok else 'SOME FAILED'}")
    print(f"  Output images   : {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()