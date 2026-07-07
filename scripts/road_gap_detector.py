"""
road_gap_detector.py
====================
RouteTREE – Road Gap Detection Module
Module 1 of 5 in the Road Healing Pipeline

PURPOSE
-------
Finds pairs of disconnected road endpoints that are geometrically close
enough to plausibly belong to the same road segment that has been
interrupted by:
    - Tree canopy occlusion
    - Building shadows
    - Vehicles
    - Clouds
    - Dense urban clutter

This module does NOT decide whether to heal the gap — that is the job of
the downstream modules (orientation_analyzer → candidate_matcher →
gap_scoring → graph_healer).

This module ONLY finds and returns *candidate* gap pairs with basic
geometric measurements attached.

POSITION IN PIPELINE
--------------------
Road Mask
    ↓
Skeleton
    ↓
Node Detection  (endpoints / junctions / turns)
    ↓
Edge Generation
    ↓
Road Graph  (NetworkX)
    ↓
>>> RoadGapDetector  <<<   ← YOU ARE HERE
    ↓
OrientationAnalyzer
    ↓
CandidateMatcher
    ↓
GapScoring
    ↓
GraphHealer

INPUT
-----
    - G         : networkx.Graph   — the road network graph
    - positions : dict             — { node_id : (x, y) }  pixel coords
    - skeleton  : np.ndarray       — binary skeleton (0/1 or 0/255)

OUTPUT
------
List of candidate gap dicts:
    {
        "start_node"           : int or str,   node id
        "end_node"             : int or str,   node id
        "start_pos"            : (x, y),
        "end_pos"              : (x, y),
        "distance"             : float,        Euclidean px distance
        "direction_difference" : float,        placeholder (filled by OrientationAnalyzer)
        "confidence"           : float,        placeholder (filled by GapScoring)
    }

USAGE
-----
    from road_gap_detector import RoadGapDetector

    detector = RoadGapDetector(search_radius=50)
    candidates = detector.find_gaps(G, positions, skeleton)
    for c in candidates:
        print(c)

DEPENDENCIES
------------
    Python   >= 3.8
    NumPy    >= 1.24
    OpenCV   >= 4.x
    NetworkX >= 3.x
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NodeID    = Any                        # whatever type the graph uses for nodes
Point     = Tuple[int, int]            # (x, y) pixel coordinate
Candidate = Dict[str, Any]             # one gap candidate record


class RoadGapDetector:
    """
    Detects candidate road gaps in a NetworkX road graph.

    A *gap* is a pair of endpoints (degree-1 nodes) that:
        1. Are NOT already connected by an edge (direct or indirect
           within the search radius).
        2. Are within ``search_radius`` Euclidean pixels of each other.
        3. Are not the same node.

    Parameters
    ----------
    search_radius : int
        Maximum Euclidean pixel distance between two endpoints for them
        to be considered a gap candidate.
        Typical values:
            30  – tight urban grid, short occlusions
            50  – standard satellite imagery at ~0.5 m/px  (default)
            75  – sparser rural roads or larger occlusions
            100 – very large cloud / canopy shadows

    min_distance : float
        Minimum distance below which two endpoints are considered
        already-connected noise rather than a real gap.  Prevents
        flagging skeleton artefacts that produced two endpoints within
        1-2 pixels of each other.  Default: 5.0 px.
    """

    def __init__(
        self,
        search_radius : int   = 50,
        min_distance  : float = 5.0,
    ) -> None:
        self.search_radius = int(search_radius)
        self.min_distance  = float(min_distance)

    # -----------------------------------------------------------------------
    # Primary public method
    # -----------------------------------------------------------------------

    def find_gaps(
        self,
        G         : nx.Graph,
        positions : Dict[NodeID, Point],
        skeleton  : np.ndarray,
    ) -> List[Candidate]:
        """
        Find all candidate road gaps in the graph.

        Steps
        -----
        1. Extract every endpoint (degree-1 node) from the graph.
        2. For every endpoint A, scan all other endpoints B within
           ``search_radius`` pixels.
        3. Skip pairs that are already connected by a direct edge.
        4. Record each valid (A, B) pair exactly once (A < B ordering
           prevents duplicates).
        5. Return sorted by distance (closest gaps first — these are
           most likely to be real occlusions rather than road ends).

        Parameters
        ----------
        G         : networkx.Graph
        positions : dict  { node_id : (x, y) }
        skeleton  : np.ndarray  binary skeleton, used for future modules
                    (stored in candidates for downstream use)

        Returns
        -------
        List[Candidate]
            Each candidate is a dict with keys:
            start_node, end_node, start_pos, end_pos,
            distance, direction_difference, confidence.
            direction_difference and confidence are set to -1.0 as
            placeholders — filled in by OrientationAnalyzer / GapScoring.
        """
        # Validate inputs
        if G is None or len(G.nodes) == 0:
            return []
        if not positions:
            return []

        # Step 1: collect endpoint node ids (degree == 1 in the road graph)
        endpoints: List[NodeID] = self._get_endpoints(G)

        if len(endpoints) < 2:
            # Need at least two endpoints to form a gap
            return []

        # Step 2-4: pairwise gap search
        candidates: List[Candidate] = self._search_pairs(G, positions, endpoints)

        # Step 5: sort by distance so closest (most likely real) gaps come first
        candidates.sort(key=lambda c: c["distance"])

        return candidates

    # -----------------------------------------------------------------------
    # Endpoint extraction
    # -----------------------------------------------------------------------

    def _get_endpoints(self, G: nx.Graph) -> List[NodeID]:
        """
        Return all nodes with degree == 1.

        In a road skeleton graph, degree-1 nodes are street dead-ends or
        points where the skeleton was interrupted (occluded).  Both are
        valid gap candidates.
        """
        return [node for node, degree in G.degree() if degree == 1]

    # -----------------------------------------------------------------------
    # Pairwise gap search
    # -----------------------------------------------------------------------

    def _search_pairs(
        self,
        G         : nx.Graph,
        positions : Dict[NodeID, Point],
        endpoints : List[NodeID],
    ) -> List[Candidate]:
        """
        For every ordered pair (A, B) of endpoints, check whether they
        qualify as a gap candidate.

        We iterate over (i, j) with i < j to avoid registering the same
        pair twice.

        Rejection criteria (any one of these rejects the pair):
            • Same node (i == j)
            • Distance < min_distance   (skeleton noise, not a real gap)
            • Distance > search_radius  (too far apart to be same road)
            • Already connected by a direct graph edge
        """
        candidates: List[Candidate] = []

        n = len(endpoints)
        r2_max = self.search_radius ** 2   # compare squared distances → no sqrt until needed
        r2_min = self.min_distance  ** 2

        for i in range(n):
            node_a = endpoints[i]
            pos_a  = positions.get(node_a)
            if pos_a is None:
                continue
            ax, ay = pos_a

            for j in range(i + 1, n):
                node_b = endpoints[j]
                pos_b  = positions.get(node_b)
                if pos_b is None:
                    continue
                bx, by = pos_b

                # Fast squared-distance check before computing sqrt
                dx, dy = bx - ax, by - ay
                dist2  = dx * dx + dy * dy

                if dist2 < r2_min:
                    # Too close — skeleton artefact, not a real gap
                    continue

                if dist2 > r2_max:
                    # Too far apart
                    continue

                # Check for existing direct edge between A and B
                if G.has_edge(node_a, node_b):
                    # Already connected — not a gap
                    continue

                # Valid gap candidate
                distance = math.sqrt(dist2)

                candidate = self._make_candidate(
                    node_a, node_b,
                    pos_a,  pos_b,
                    distance,
                )
                candidates.append(candidate)

        return candidates

    # -----------------------------------------------------------------------
    # Candidate record factory
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_candidate(
        node_a   : NodeID,
        node_b   : NodeID,
        pos_a    : Point,
        pos_b    : Point,
        distance : float,
    ) -> Candidate:
        """
        Build a single candidate gap record.

        ``direction_difference`` and ``confidence`` are set to -1.0 here
        as explicit placeholders.  They will be filled in by:
            - OrientationAnalyzer → direction_difference
            - GapScoring          → confidence

        This makes it easy to detect in downstream modules whether a
        field has been populated or is still unprocessed.
        """
        return {
            "start_node"           : node_a,
            "end_node"             : node_b,
            "start_pos"            : pos_a,
            "end_pos"              : pos_b,
            "distance"             : round(distance, 3),
            "direction_difference" : -1.0,   # filled by OrientationAnalyzer
            "confidence"           : -1.0,   # filled by GapScoring
        }

    # -----------------------------------------------------------------------
    # Utility: get endpoints from node list (alternative entry point)
    # -----------------------------------------------------------------------

    @staticmethod
    def get_endpoints_from_graph(G: nx.Graph) -> List[NodeID]:
        """
        Public static helper — returns all degree-1 nodes.
        Useful for external modules that just need the endpoint list
        without running the full gap search.
        """
        return [node for node, deg in G.degree() if deg == 1]

    # -----------------------------------------------------------------------
    # Summary / diagnostics
    # -----------------------------------------------------------------------

    @staticmethod
    def summarize(candidates: List[Candidate]) -> None:
        """
        Print a human-readable summary of detected gap candidates.

        Parameters
        ----------
        candidates : List[Candidate]
            Output from find_gaps().
        """
        print("=" * 50)
        print(f"Road Gap Detector — {len(candidates)} candidate gap(s) found")
        print("=" * 50)

        if not candidates:
            print("  No gaps detected.")
            return

        for i, c in enumerate(candidates):
            print(
                f"  [{i+1:3d}]  "
                f"{str(c['start_node']):>6} → {str(c['end_node']):<6}  "
                f"dist={c['distance']:6.1f}px  "
                f"start={c['start_pos']}  "
                f"end={c['end_pos']}"
            )

        distances = [c["distance"] for c in candidates]
        print("-" * 50)
        print(f"  Min distance : {min(distances):.1f} px")
        print(f"  Max distance : {max(distances):.1f} px")
        print(f"  Avg distance : {sum(distances)/len(distances):.1f} px")

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def visualize(
        self,
        image      : np.ndarray,
        positions  : Dict[NodeID, Point],
        candidates : List[Candidate],
        G          : Optional[nx.Graph] = None,
        output_path: Optional[str]      = None,
    ) -> np.ndarray:
        """
        Draw detected gap candidates onto a copy of *image*.

        Visual legend
        -------------
            Green circles   : all graph nodes
            Blue circles    : endpoint nodes (degree-1)
            Yellow dashed   : candidate gap lines
            Red text        : distance label on each gap

        Parameters
        ----------
        image       : np.ndarray   BGR image to annotate (H, W, 3)
        positions   : dict         { node_id : (x, y) }
        candidates  : List[Candidate]
        G           : networkx.Graph  (optional, for drawing existing edges)
        output_path : str             (optional) save path, e.g.
                      "D:/Projects/RouteTREE/outputs/comparisons/gaps.png"

        Returns
        -------
        np.ndarray  annotated BGR image
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw existing graph edges in green
        if G is not None:
            for u, v in G.edges():
                pu = positions.get(u)
                pv = positions.get(v)
                if pu and pv:
                    cv2.line(canvas, pu, pv, (0, 200, 0), 1)

        # Draw all nodes as small green circles
        for node, pos in positions.items():
            cv2.circle(canvas, pos, 3, (0, 200, 0), -1)

        # Draw endpoint nodes as blue circles
        if G is not None:
            for node, deg in G.degree():
                if deg == 1 and node in positions:
                    cv2.circle(canvas, positions[node], 5, (255, 100, 0), -1)

        # Draw each candidate gap as a yellow dashed line
        for c in candidates:
            p1 = c["start_pos"]
            p2 = c["end_pos"]

            # Draw dashed line by splitting into segments
            self._draw_dashed_line(canvas, p1, p2, (0, 255, 255), thickness=1, dash_len=6)

            # Distance label at midpoint
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                canvas,
                f"{c['distance']:.0f}px",
                (mid_x + 3, mid_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            # Mark gap endpoints
            cv2.circle(canvas, p1, 4, (0, 255, 255), -1)
            cv2.circle(canvas, p2, 4, (0, 255, 255), -1)

        # Legend
        self._draw_legend(canvas)

        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"  Visualization saved → {output_path}")

        return canvas

    # -----------------------------------------------------------------------
    # Internal drawing helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _draw_dashed_line(
        image    : np.ndarray,
        p1       : Point,
        p2       : Point,
        color    : Tuple[int, int, int],
        thickness: int = 1,
        dash_len : int = 8,
    ) -> None:
        """
        Draw a dashed line from p1 to p2.

        Splits the line into segments of length ``dash_len`` pixels,
        drawing every other segment.
        """
        x1, y1 = p1
        x2, y2 = p2
        total  = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if total < 1:
            return

        # Direction unit vector
        ux = (x2 - x1) / total
        uy = (y2 - y1) / total

        dist  = 0.0
        draw  = True   # alternate draw / skip
        while dist < total:
            seg_end = min(dist + dash_len, total)
            if draw:
                sx1 = int(x1 + ux * dist)
                sy1 = int(y1 + uy * dist)
                sx2 = int(x1 + ux * seg_end)
                sy2 = int(y1 + uy * seg_end)
                cv2.line(image, (sx1, sy1), (sx2, sy2), color, thickness)
            dist += dash_len
            draw  = not draw

    @staticmethod
    def _draw_legend(canvas: np.ndarray) -> None:
        """Draw a small legend in the top-left corner of the canvas."""
        items = [
            ((0, 200, 0),   "Existing road"),
            ((255, 100, 0), "Endpoints"),
            ((0, 255, 255), "Gap candidates"),
        ]
        x, y = 8, 16
        for color, label in items:
            cv2.rectangle(canvas, (x, y - 8), (x + 12, y + 2), color, -1)
            cv2.putText(canvas, label, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
