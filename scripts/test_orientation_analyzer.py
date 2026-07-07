"""
test_orientation_analyzer.py
=============================
Test for OrientationAnalyzer — Module 2 of the RouteTREE healing pipeline.

HOW TO RUN
----------
    cd D:/Projects/RouteTREE/scripts
    python test_orientation_analyzer.py

    OR with a specific mask:
    python test_orientation_analyzer.py --mask path/to/your_mask.png

WHAT THIS DOES
--------------
    1. Loads a road mask PNG.
    2. Runs the full pipeline to get skeleton + graph.
    3. Extracts all endpoints.
    4. Runs OrientationAnalyzer on every endpoint.
    5. Prints direction vectors and angles per endpoint.
    6. Saves a visualization showing direction arrows.
    7. Runs synthetic unit tests (horizontal / vertical / diagonal roads).

OUTPUT FILES
------------
    outputs/comparisons/orientation_all_endpoints.png
    outputs/comparisons/orientation_synthetic.png
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

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from graph_utils          import GraphUtils
from node_detector        import NodeDetector
from edge_generator       import EdgeGenerator
from network_builder      import NetworkBuilder
from label_generator      import LabelGenerator
from road_gap_detector    import RoadGapDetector
from orientation_analyzer import OrientationAnalyzer

# ---------------------------------------------------------------------------
# Default mask
# ---------------------------------------------------------------------------
DEFAULT_MASK = os.path.join(
    PROJECT_ROOT, "datasets", "deepglobe", "masks", "road_mask.png"
)

PASS = "✓"
FAIL = "✗"


# ---------------------------------------------------------------------------
# Pipeline helper (same as test_road_gap_detector.py)
# ---------------------------------------------------------------------------

def build_graph_from_mask(mask: np.ndarray):
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
# Synthetic tests
# ---------------------------------------------------------------------------

def test_synthetic():
    """
    Run OrientationAnalyzer on hand-crafted skeletons where the
    expected direction is known exactly.
    """
    print(f"\n{'='*60}")
    print("  Synthetic unit tests")
    print(f"{'='*60}")

    analyzer = OrientationAnalyzer(walk_length=20)
    results  = []

    # ------------------------------------------------------------------
    # Test 1: Horizontal road going RIGHT
    # Endpoint at (10, 50), road goes east (to x=80)
    # Expected angle ≈ 0° (East)
    # ------------------------------------------------------------------
    skel1 = np.zeros((100, 100), dtype=np.uint8)
    skel1[50, 10:80] = 1
    res1  = analyzer.estimate_direction(skel1, endpoint=(10, 50))
    ang1  = res1["angle_deg"]
    ok1   = res1["valid"] and abs(ang1 - 0.0) <= 15.0
    print(f"\n  Test 1 – Horizontal road (East endpoint)")
    print(f"    Expected angle : ~0° (East)")
    print(f"    Got angle      : {ang1:.1f}°")
    print(f"    Valid          : {res1['valid']}")
    print(f"    Result         : {PASS if ok1 else FAIL}")
    results.append(ok1)

    # ------------------------------------------------------------------
    # Test 2: Vertical road going DOWN
    # Endpoint at (50, 10), road goes south (to y=80)
    # Expected angle ≈ 270° (South in image coords)
    # ------------------------------------------------------------------
    skel2 = np.zeros((100, 100), dtype=np.uint8)
    skel2[10:80, 50] = 1
    res2  = analyzer.estimate_direction(skel2, endpoint=(50, 10))
    ang2  = res2["angle_deg"]
    # South in image coords = 270°, but allow 90° too (direction might flip)
    ok2   = res2["valid"] and (abs(ang2 - 270.0) <= 15.0 or abs(ang2 - 90.0) <= 15.0)
    print(f"\n  Test 2 – Vertical road (South endpoint)")
    print(f"    Expected angle : ~270° or ~90°")
    print(f"    Got angle      : {ang2:.1f}°")
    print(f"    Valid          : {res2['valid']}")
    print(f"    Result         : {PASS if ok2 else FAIL}")
    results.append(ok2)

    # ------------------------------------------------------------------
    # Test 3: 45° diagonal road (NE direction)
    # Endpoint at (10, 90), road goes toward (80, 20)
    # Expected angle ≈ 45° (Northeast)
    # ------------------------------------------------------------------
    skel3 = np.zeros((100, 100), dtype=np.uint8)
    for i in range(70):
        skel3[90 - i, 10 + i] = 1
    res3  = analyzer.estimate_direction(skel3, endpoint=(10, 90))
    ang3  = res3["angle_deg"]
    ok3   = res3["valid"] and (abs(ang3 - 45.0) <= 20.0 or abs(ang3 - 225.0) <= 20.0)
    print(f"\n  Test 3 – Diagonal road (NE)")
    print(f"    Expected angle : ~45°")
    print(f"    Got angle      : {ang3:.1f}°")
    print(f"    Valid          : {res3['valid']}")
    print(f"    Result         : {PASS if ok3 else FAIL}")
    results.append(ok3)

    # ------------------------------------------------------------------
    # Test 4: Two endpoints facing each other (interrupted road)
    # Road at y=50, x=10→40 (gap) x=60→90
    # Left endpoint (10,50) should point East (~0°)
    # Right endpoint (90,50) should point West (~180°)
    # Their angles should differ by ~180° → facing each other
    # ------------------------------------------------------------------
    skel4 = np.zeros((100, 100), dtype=np.uint8)
    skel4[50, 10:40] = 1    # left segment
    skel4[50, 60:90] = 1    # right segment (gap in middle)
    res4a = analyzer.estimate_direction(skel4, endpoint=(10, 50))
    res4b = analyzer.estimate_direction(skel4, endpoint=(90, 50))
    ang4a = res4a["angle_deg"]
    ang4b = res4b["angle_deg"]
    diff4 = OrientationAnalyzer.angle_difference(ang4a, ang4b)
    # Facing each other means angle difference is close to 180°
    ok4   = res4a["valid"] and res4b["valid"] and diff4 >= 140.0
    print(f"\n  Test 4 – Interrupted road (two endpoints facing each other)")
    print(f"    Left endpoint angle  : {ang4a:.1f}°")
    print(f"    Right endpoint angle : {ang4b:.1f}°")
    print(f"    Angle difference     : {diff4:.1f}° (expected ~180°)")
    print(f"    Result               : {PASS if ok4 else FAIL}")
    results.append(ok4)

    # ------------------------------------------------------------------
    # Test 5: Two endpoints going SAME direction (NOT a gap)
    # Road at y=50, x=10→40 and y=50, x=10→40 shifted — both pointing East
    # ------------------------------------------------------------------
    skel5 = np.zeros((100, 100), dtype=np.uint8)
    skel5[50, 10:40] = 1    # segment A
    skel5[60, 10:40] = 1    # segment B (parallel, same direction)
    res5a = analyzer.estimate_direction(skel5, endpoint=(39, 50))
    res5b = analyzer.estimate_direction(skel5, endpoint=(39, 60))
    ang5a = res5a["angle_deg"]
    ang5b = res5b["angle_deg"]
    diff5 = OrientationAnalyzer.angle_difference(ang5a, ang5b)
    # Same direction means angle difference is close to 0°
    ok5   = res5a["valid"] and res5b["valid"] and diff5 <= 30.0
    print(f"\n  Test 5 – Parallel roads (same direction, NOT a gap)")
    print(f"    Segment A angle : {ang5a:.1f}°")
    print(f"    Segment B angle : {ang5b:.1f}°")
    print(f"    Angle difference: {diff5:.1f}° (expected ~0°)")
    print(f"    Result          : {PASS if ok5 else FAIL}")
    results.append(ok5)

    # ------------------------------------------------------------------
    # Test 6: Invalid endpoint (isolated pixel)
    # ------------------------------------------------------------------
    skel6 = np.zeros((100, 100), dtype=np.uint8)
    skel6[50, 50] = 1    # single isolated pixel
    res6  = analyzer.estimate_direction(skel6, endpoint=(50, 50))
    ok6   = not res6["valid"]
    print(f"\n  Test 6 – Isolated pixel (should be invalid)")
    print(f"    Valid  : {res6['valid']} (expected False)")
    print(f"    Result : {PASS if ok6 else FAIL}")
    results.append(ok6)

    # ------------------------------------------------------------------
    # Synthetic visualization
    # ------------------------------------------------------------------
    canvas = np.zeros((100, 300, 3), dtype=np.uint8)

    # Draw skel1, skel2, skel3 side by side
    for col_offset, skel, res in [
        (0,   skel1, res1),
        (100, skel2, res2),
        (200, skel3, res3),
    ]:
        # Paste skeleton
        bg = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
        canvas[:, col_offset:col_offset+100] = bg
        # Draw arrow — adjust endpoint x for canvas offset
        ep_shifted = (res["endpoint"][0] + col_offset, res["endpoint"][1])
        if res["valid"]:
            ux, uy = res["unit_vector"]
            tip = (
                int(round(ep_shifted[0] + ux * 25)),
                int(round(ep_shifted[1] + uy * 25)),
            )
            cv2.arrowedLine(canvas, ep_shifted, tip, (0, 165, 255), 2,
                            tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.circle(canvas, ep_shifted, 3, (0, 255, 0), -1)

    out_path = os.path.join(OUTPUT_DIR, "orientation_synthetic.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n  Synthetic visualization saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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

    print("Building graph ...")
    G, positions, skeleton, all_points, nodes_dict = build_graph_from_mask(mask)

    # Get endpoint coordinates from the graph
    detector_gap  = RoadGapDetector()
    ep_labels     = detector_gap.get_endpoints_from_graph(G)
    ep_coords     = [positions[lbl] for lbl in ep_labels if lbl in positions]

    print(f"  Endpoints to analyze : {len(ep_coords)}")

    # Run analyzer
    analyzer = OrientationAnalyzer(walk_length=20)
    results  = analyzer.analyze_all(skeleton, ep_coords)

    # Print per-endpoint results
    print(f"\n  {'Endpoint':>15}  {'Angle':>8}  {'Unit Vector':>22}  Valid")
    print(f"  {'-'*15}  {'-'*8}  {'-'*22}  -----")
    for ep, res in results.items():
        ux, uy = res["unit_vector"]
        print(
            f"  {str(ep):>15}  "
            f"{res['angle_deg']:>7.1f}°  "
            f"({ux:+.3f}, {uy:+.3f})        "
            f"{'yes' if res['valid'] else 'NO'}"
        )

    # Also test angle_difference between gap pairs
    gap_detector = RoadGapDetector(search_radius=75)
    candidates   = gap_detector.find_gaps(G, positions, skeleton)

    if candidates:
        print(f"\n  Angle differences for {len(candidates)} gap candidates (r=75px):")
        print(f"  {'Pair':>10}  {'Angle A':>8}  {'Angle B':>8}  {'Diff':>8}  Facing?")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  -------")
        for c in candidates:
            pa = c["start_pos"]
            pb = c["end_pos"]
            ra = results.get(pa)
            rb = results.get(pb)
            if ra and rb and ra["valid"] and rb["valid"]:
                diff = OrientationAnalyzer.angle_difference(
                    ra["angle_deg"], rb["angle_deg"]
                )
                facing = diff >= 120.0
                print(
                    f"  {str(pa):>10} → {str(pb):<10}  "
                    f"{ra['angle_deg']:>7.1f}°  "
                    f"{rb['angle_deg']:>7.1f}°  "
                    f"{diff:>7.1f}°  "
                    f"{'YES' if facing else 'no'}"
                )

    # Visualization
    canvas = vis_base.copy()
    canvas = analyzer.visualize_all(
        canvas, results,
        output_path=os.path.join(OUTPUT_DIR, "orientation_all_endpoints.png")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test OrientationAnalyzer")
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