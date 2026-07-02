"""
node_detector.py
================
RouteTREE – Road Network Node Detection
Author  : RouteTREE Research Pipeline
Version : 2.0  (complete redesign)

PURPOSE
-------
Detects three semantically meaningful node types on a thinned binary road
skeleton produced by a DeepLabV3+ segmentation model:

  • Endpoints  – pixels where exactly one road branch terminates
  • Junctions  – pixels where three or more road branches meet
                 (T-junction, Y-junction, X-junction / crossroads)
  • Turns      – pixels where a road bends significantly, i.e. the local
                 direction vector changes by more than a configurable angle
                 threshold along a single branch

The output dictionary is directly consumed by edge_generator.py,
graph_utils.py, network_builder.py, and label_generator.py without any
format change from the previous API.

WHY THIS REDESIGN IS NECESSARY
--------------------------------
The naive approach of measuring the skeleton-pixel angle at every pixel
fails because a discrete raster skeleton made of 8-connected pixels
produces a different "angle" at every diagonal step even on a perfectly
straight road.  This floods the graph with hundreds of spurious turn nodes.

This implementation instead:
  1. Identifies junction *regions* (clusters of high-connectivity pixels)
     and collapses each cluster to its centroid.
  2. Traces each skeleton *branch* between junction/endpoint seed points
     as an ordered list of pixels using 8-connected DFS chain tracing.
  3. Applies the Douglas-Peucker polyline simplification algorithm to each
     branch, reducing hundreds of raster points to a small set of
     geometrically significant vertices.
  4. Examines only the *interior* vertices of the simplified polyline:
     if the turning angle there exceeds TURN_ANGLE_THRESHOLD_DEG, that
     vertex becomes a turn node.
  5. Suppresses any turn node that falls within EXCLUSION_RADIUS pixels
     of an already-accepted endpoint or junction.
  6. Performs a final global merge to collapse any remaining near-duplicate
     nodes within MERGE_RADIUS pixels of each other.

This strategy is mathematically sound: it measures geometric direction
change on a smooth polyline approximation, not on noisy raster pixels.

ALGORITHM OVERVIEW
------------------

  Phase 1 – Endpoint & Junction seed detection
    ┌───────────────────────────────────────┐
    │  For every foreground pixel p in the  │
    │  skeleton, count its 8-connected      │
    │  foreground neighbours (valence).     │
    │  valence == 1  →  endpoint candidate  │
    │  valence >= 3  →  junction candidate  │
    └───────────────────────────────────────┘

  Phase 2 – Junction region merging
    Adjacent junction candidates are connected components.  Replace each
    connected component with its centroid (integer coordinates).

  Phase 3 – Branch tracing
    Walk the skeleton with DFS starting from each endpoint and junction
    seed, following 8-connected foreground pixels that have not yet been
    visited.  Record the ordered pixel sequence of every branch.  Branches
    that are too short to carry a turn (< MIN_BRANCH_LENGTH) are skipped.

  Phase 4 – Turn detection via Douglas-Peucker + angle threshold
    Simplify each branch with Douglas-Peucker (epsilon = DP_EPSILON).
    For every interior vertex of the simplified polyline, compute the
    angle between the incoming and outgoing direction vectors.
    If the angle > TURN_ANGLE_THRESHOLD_DEG → candidate turn node at
    the corresponding skeleton pixel (mapped back from the simplified
    polyline index to the original branch pixel list).

  Phase 5 – Exclusion zone filtering
    Remove any turn candidate within EXCLUSION_RADIUS pixels of any
    endpoint or junction centroid.

  Phase 6 – Global node deduplication
    Merge any two nodes of any type within MERGE_RADIUS of each other,
    keeping the one with higher local valence.

DEPENDENCIES
------------
  Python  ≥ 3.8
  OpenCV  ≥ 4.x   (cv2)
  NumPy   ≥ 1.24

  No scikit-image, no Shapely, no external GIS libraries.

USAGE
-----
  from node_detector import NodeDetector

  detector = NodeDetector(
      turn_angle_threshold_deg = 35.0,
      dp_epsilon               = 3.0,
      exclusion_radius         = 12,
      merge_radius             = 8,
      min_branch_length        = 15,
  )

  nodes  = detector.detect_nodes(skeleton_binary_uint8)
  canvas = detector.draw_nodes(bgr_image, nodes)
  count  = detector.node_count(nodes)
  all_pts = detector.get_all_nodes(nodes)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases (for readability / IDE support)
# ---------------------------------------------------------------------------
Point      = Tuple[int, int]          # (col, row) = (x, y) convention
NodeDict   = Dict[str, List[Point]]   # {"endpoints": [...], ...}


# ---------------------------------------------------------------------------
# 8-connectivity neighbour offsets
# ---------------------------------------------------------------------------
_N8_OFFSETS: List[Tuple[int, int]] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


# ===========================================================================
# NodeDetector
# ===========================================================================
class NodeDetector:
    """
    Detects endpoints, junctions, and turn nodes on a binary road skeleton.

    Parameters
    ----------
    turn_angle_threshold_deg : float
        Minimum turning angle (in degrees) that is considered a real road
        bend.  Angles below this are treated as straight road and do NOT
        produce a turn node.  Default 35°; raise it to get fewer turns,
        lower it to get more.  Recommended range: 25°–50°.

    dp_epsilon : float
        Douglas-Peucker simplification tolerance in pixels.  Controls how
        aggressively the branch polyline is simplified before angle
        measurement.  Larger values → smoother polyline → fewer turns
        detected.  Default 3.0 px (works well for 512-px and 1024-px
        masks).  Scale proportionally for 2048-px masks (e.g. 5.0–6.0).

    exclusion_radius : int
        Radius in pixels around each endpoint / junction within which
        turn nodes are suppressed.  Prevents spurious turns caused by the
        raster artefacts that always appear at branch roots.  Default 12.

    merge_radius : int
        Any two accepted nodes within this Euclidean pixel distance are
        merged into one (keeping the first in the sorted list).  Default 8.

    min_branch_length : int
        Branches shorter than this pixel count are discarded entirely
        before turn detection.  Very short branches are usually noise from
        the skeletonization step and never carry meaningful turns.
        Default 15.
    """

    def __init__(
        self,
        turn_angle_threshold_deg: float = 35.0,
        dp_epsilon               : float = 3.0,
        exclusion_radius         : int   = 12,
        merge_radius             : int   = 8,
        min_branch_length        : int   = 15,
    ) -> None:
        self.turn_angle_threshold_deg = float(turn_angle_threshold_deg)
        self.dp_epsilon               = float(dp_epsilon)
        self.exclusion_radius         = int(exclusion_radius)
        self.merge_radius             = int(merge_radius)
        self.min_branch_length        = int(min_branch_length)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect_nodes(self, skeleton: np.ndarray) -> NodeDict:
        """
        Full node detection pipeline.

        Parameters
        ----------
        skeleton : np.ndarray
            Binary skeleton image, dtype uint8, shape (H, W).
            Foreground (road) pixels must be > 0.

        Returns
        -------
        NodeDict
            {
              "endpoints"  : List[Tuple[int,int]],   # (x, y) = (col, row)
              "junctions"  : List[Tuple[int,int]],
              "turns"      : List[Tuple[int,int]],
            }
            All coordinates are in (x, y) = (column, row) order so they
            can be passed directly to cv2 drawing functions and to
            edge_generator.py without further conversion.
        """
        skel = self._to_binary(skeleton)
        if skel is None or skel.sum() == 0:
            return {"endpoints": [], "junctions": [], "turns": []}

        # Phase 1: per-pixel valence map
        valence_map = self._compute_valence(skel)

        # Phase 2: raw endpoint / junction pixel sets
        ep_pixels, jn_pixels = self._classify_pixels(skel, valence_map)

        # Phase 3: merge junction clusters → centroid points
        junctions: List[Point] = self._merge_junction_clusters(jn_pixels, skel)

        # Phase 4: snap endpoint pixels to clean single-pixel representatives
        endpoints: List[Point] = self._clean_endpoints(ep_pixels)

        # Phase 5: trace branches, detect turns via DP + angle
        all_seeds: List[Point] = endpoints + junctions
        turns: List[Point] = self._detect_turns(skel, all_seeds)

        # Phase 6: suppress turns near endpoints / junctions
        anchor_pts: List[Point] = endpoints + junctions
        turns = self._suppress_near_anchors(turns, anchor_pts, self.exclusion_radius)

        # Phase 7: global deduplication / merge across all node types
        endpoints, junctions, turns = self._global_merge(endpoints, junctions, turns)

        return {
            "endpoints" : endpoints,
            "junctions" : junctions,
            "turns"     : turns,
        }

    def draw_nodes(
        self,
        image : np.ndarray,
        nodes : NodeDict,
    ) -> np.ndarray:
        """
        Draw detected nodes onto a copy of *image*.

        Colour convention
        -----------------
          Endpoints  → green   (0, 255, 0)
          Junctions  → red     (0, 0, 255)
          Turns      → yellow  (0, 255, 255)

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape (H, W, 3).  Unmodified; a copy is returned.
        nodes : NodeDict
            Output from detect_nodes().

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        style: List[Tuple[str, Tuple[int,int,int], int]] = [
            ("endpoints",  (0, 255,   0), 5),   # green  filled circle
            ("junctions",  (0,   0, 255), 7),   # red    filled circle
            ("turns",      (0, 255, 255), 4),   # yellow filled circle
        ]

        for key, colour, radius in style:
            for (x, y) in nodes.get(key, []):
                cv2.circle(canvas, (int(x), int(y)), radius, colour, -1)
                cv2.circle(canvas, (int(x), int(y)), radius + 1, (0,0,0), 1)

        return canvas

    def get_all_nodes(self, nodes: NodeDict) -> List[Point]:
        """
        Return a flat list of every node across all categories.

        Order: endpoints → junctions → turns.

        Parameters
        ----------
        nodes : NodeDict
            Output from detect_nodes().

        Returns
        -------
        List[Point]
        """
        return (
            list(nodes.get("endpoints", []))
            + list(nodes.get("junctions", []))
            + list(nodes.get("turns", []))
        )

    def node_count(self, nodes: NodeDict) -> Dict[str, int]:
        """
        Return per-category and total node counts.

        Parameters
        ----------
        nodes : NodeDict
            Output from detect_nodes().

        Returns
        -------
        Dict[str, int]
            {
              "endpoints" : int,
              "junctions" : int,
              "turns"     : int,
              "total"     : int,
            }
        """
        eps = len(nodes.get("endpoints", []))
        jns = len(nodes.get("junctions", []))
        tns = len(nodes.get("turns",     []))
        return {
            "endpoints" : eps,
            "junctions" : jns,
            "turns"     : tns,
            "total"     : eps + jns + tns,
        }

    # -----------------------------------------------------------------------
    # Phase helpers
    # -----------------------------------------------------------------------

    # -- Preprocessing -------------------------------------------------------

    @staticmethod
    def _to_binary(skeleton: np.ndarray) -> np.ndarray | None:
        """
        Ensure the skeleton is a strict uint8 binary image (0 or 1).
        Accepts 0/255 or 0/1 encodings.  Returns None if input is invalid.
        """
        if skeleton is None or skeleton.size == 0:
            return None
        skel = skeleton.astype(np.uint8)
        if skel.max() > 1:
            _, skel = cv2.threshold(skel, 127, 1, cv2.THRESH_BINARY)
        return skel

    # -- Phase 1: valence map ------------------------------------------------

    @staticmethod
    def _compute_valence(skel: np.ndarray) -> np.ndarray:
        """
        For every foreground pixel, count the number of foreground
        8-connected neighbours.  Background pixels are left at 0.

        This is the "valence" (graph-theoretic degree) of each pixel
        within the skeleton raster graph.

        Uses a 2-D convolution approach for speed on large images:
        convolve the binary skeleton with the 3×3 all-ones kernel, then
        subtract 1 (so the pixel itself is not counted).
        """
        kernel   = np.ones((3, 3), dtype=np.uint8)
        conv     = cv2.filter2D(skel.astype(np.float32), -1, kernel.astype(np.float32))
        valence  = (conv.astype(np.int32) - skel.astype(np.int32))
        valence[skel == 0] = 0   # zero out background
        return valence.astype(np.int32)

    # -- Phase 2: classify pixels --------------------------------------------

    @staticmethod
    def _classify_pixels(
        skel       : np.ndarray,
        valence_map: np.ndarray,
    ) -> Tuple[List[Point], List[Point]]:
        """
        Classify each foreground pixel as endpoint or junction based on
        its valence.

        Valence == 1  →  endpoint  (one neighbour: branch terminates here)
        Valence >= 3  →  junction  (three or more branches meet)

        Valence == 2 pixels are ordinary straight-road pixels; they will
        be visited during branch tracing but are not nodes by themselves
        (turn detection handles bends on these segments).

        Returns
        -------
        ep_pixels : List[Point]   raw endpoint pixel coordinates
        jn_pixels : List[Point]   raw junction-candidate pixel coordinates
        """
        rows_ep, cols_ep = np.where((skel == 1) & (valence_map == 1))
        ep_pixels: List[Point] = [(int(c), int(r)) for r, c in zip(rows_ep, cols_ep)]

        rows_jn, cols_jn = np.where((skel == 1) & (valence_map >= 3))
        jn_pixels: List[Point] = [(int(c), int(r)) for r, c in zip(rows_jn, cols_jn)]

        return ep_pixels, jn_pixels

    # -- Phase 3: junction cluster merging -----------------------------------

    def _merge_junction_clusters(
        self,
        jn_pixels : List[Point],
        skel      : np.ndarray,
    ) -> List[Point]:
        """
        Real-world junctions in a rasterised skeleton are not single pixels
        but *clusters* of adjacent high-valence pixels.  A T-junction can
        span a 3×3 or 4×4 pixel region; an X-junction can span even more.

        Algorithm:
          1.  Paint junction pixels onto a blank mask.
          2.  Use OpenCV connected components to identify each cluster.
          3.  Replace each cluster with the centroid (rounded to integer).

        Parameters
        ----------
        jn_pixels : List[Point]
            Raw junction-candidate pixels from _classify_pixels().
        skel : np.ndarray
            Binary skeleton (used only for shape/bounds).

        Returns
        -------
        List[Point]
            One (x, y) centroid per merged junction cluster.
        """
        if not jn_pixels:
            return []

        h, w  = skel.shape
        jmask = np.zeros((h, w), dtype=np.uint8)
        for (x, y) in jn_pixels:
            jmask[y, x] = 1

        # Dilate slightly so junction pixels that touch diagonally
        # (very common in 8-connected skeletons) are connected.
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        jmask_d  = cv2.dilate(jmask, kernel, iterations=1)

        n_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(
            jmask_d, connectivity=8
        )

        junctions: List[Point] = []
        for label_idx in range(1, n_labels):   # skip background (label 0)
            cx = int(round(centroids[label_idx, 0]))
            cy = int(round(centroids[label_idx, 1]))
            # Clamp to image bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            junctions.append((cx, cy))

        return junctions

    # -- Phase 4: endpoint clean-up ------------------------------------------

    @staticmethod
    def _clean_endpoints(ep_pixels: List[Point]) -> List[Point]:
        """
        Endpoint candidates from valence==1 are already single pixels,
        so they rarely need merging.  However, if the skeletonizer
        produces a tiny 2-px spur that looks like two adjacent endpoints,
        we deduplicate any pair within 3 pixels of each other.
        """
        if not ep_pixels:
            return []

        merged: List[Point] = []
        used = [False] * len(ep_pixels)

        for i, p in enumerate(ep_pixels):
            if used[i]:
                continue
            cluster = [p]
            for j in range(i + 1, len(ep_pixels)):
                if not used[j] and _euclidean(p, ep_pixels[j]) <= 3.0:
                    cluster.append(ep_pixels[j])
                    used[j] = True
            used[i] = True
            cx = int(round(sum(q[0] for q in cluster) / len(cluster)))
            cy = int(round(sum(q[1] for q in cluster) / len(cluster)))
            merged.append((cx, cy))

        return merged

    # -- Phase 5: branch tracing and turn detection --------------------------

    def _detect_turns(
        self,
        skel  : np.ndarray,
        seeds : List[Point],
    ) -> List[Point]:
        """
        Main turn-detection routine.

        Step A – Trace every skeleton branch
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Starting from each seed (endpoint or junction), perform a DFS walk
        along the skeleton, recording the pixel sequence of each branch
        (an unbroken chain of 8-connected foreground pixels between two
        seeds).  This gives us the actual geometric path of each road
        segment as a polyline.

        Step B – Simplify with Douglas-Peucker
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Apply the Douglas-Peucker algorithm to each branch's pixel list.
        This transforms ~500 raster points on a curvy road into 5–10
        key geometric vertices, discarding all intermediate points that
        lie within dp_epsilon pixels of the simplified polyline.

        Why DP is the right tool here:
          • DP is O(n log n) on average and handles arbitrary point lists.
          • It preserves the geometrically significant bends while removing
            noise from raster quantisation.
          • The simplified result is a true polyline, so direction vectors
            between consecutive vertices are stable and well-defined.

        Step C – Measure turning angles at interior DP vertices
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        For each interior vertex v_i of the simplified polyline (i.e. not
        the first or last vertex), compute the angle between:
            incoming vector : v_{i-1} → v_i
            outgoing vector : v_i → v_{i+1}

        The *turning angle* is 180° − (angle between the two vectors),
        i.e. it measures the deviation from straight travel.

        If the turning angle ≥ turn_angle_threshold_deg, the corresponding
        pixel in the original branch is marked as a turn candidate.

        Parameters
        ----------
        skel  : np.ndarray   binary skeleton (0/1)
        seeds : List[Point]  endpoint + junction centroids

        Returns
        -------
        List[Point]   raw turn-candidate locations (before exclusion/merge)
        """
        h, w        = skel.shape
        seed_set    = set(seeds)
        visited     = np.zeros((h, w), dtype=bool)

        # Mark seed pixels as visited so they act as branch terminators
        for (x, y) in seeds:
            if 0 <= y < h and 0 <= x < w:
                visited[y, x] = True

        turn_candidates: List[Point] = []

        for seed in seeds:
            sx, sy = seed
            if not (0 <= sy < h and 0 <= sx < w):
                continue

            # Examine each 8-connected neighbour of the seed as a
            # potential branch start
            for (dx, dy) in _N8_OFFSETS:
                nx, ny = sx + dx, sy + dy
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if skel[ny, nx] == 0 or visited[ny, nx]:
                    continue

                # Trace branch from (nx, ny) until we hit another seed
                branch = self._trace_branch(skel, seed, (nx, ny), visited, seed_set)

                if len(branch) < self.min_branch_length:
                    # Too short to carry a meaningful turn
                    continue

                # Simplify the branch with Douglas-Peucker.
                #
                # Adaptive epsilon: a fixed epsilon works well on short
                # branches but is too tight on long ones (259-px branches
                # simplified at ε=3 still yield 9 vertices, causing the
                # raster quantisation noise to masquerade as bends).
                # Scale epsilon proportionally to branch length so that
                # the ratio of retained vertices stays stable regardless
                # of branch length.  The 0.03 factor means "keep one
                # vertex per ~33 pixels of road" which empirically gives
                # 1-3 interior vertices per smooth curve.
                adaptive_eps = max(self.dp_epsilon, len(branch) * 0.03)
                simplified_indices = _douglas_peucker(branch, adaptive_eps)

                # Check angle at each interior DP vertex
                pts = simplified_indices  # list of indices into branch[]
                for k in range(1, len(pts) - 1):
                    p_prev = branch[pts[k - 1]]
                    p_curr = branch[pts[k    ]]
                    p_next = branch[pts[k + 1]]

                    angle = _turning_angle_deg(p_prev, p_curr, p_next)
                    if angle >= self.turn_angle_threshold_deg:
                        turn_candidates.append(p_curr)

        return turn_candidates

    @staticmethod
    def _trace_branch(
        skel     : np.ndarray,
        start    : Point,
        first    : Point,
        visited  : np.ndarray,
        seed_set : set,
    ) -> List[Point]:
        """
        Trace a single branch of the skeleton starting from pixel *first*
        (the first non-seed pixel adjacent to *start*).

        The trace follows 8-connected foreground pixels in a DFS manner.
        It stops when:
          (a) there are no unvisited foreground neighbours, OR
          (b) the current pixel is in seed_set (we've reached another node)

        Pixels along the branch are marked visited as we go, preventing
        the same branch from being traced twice in the opposite direction.

        Returns
        -------
        List[Point]
            Ordered pixel sequence of the branch INCLUDING the start seed
            pixel and the terminating pixel, so direction vectors at the
            very start and end of the branch are well-defined.
        """
        h, w    = skel.shape
        branch  = [start, first]
        visited[first[1], first[0]] = True

        current = first
        while True:
            cx, cy = current

            # Collect unvisited foreground neighbours (potential next steps)
            nexts: List[Point] = []
            for (dx, dy) in _N8_OFFSETS:
                nx, ny = cx + dx, cy + dy
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] == 1:
                    if not visited[ny, nx]:
                        nexts.append((nx, ny))

            if not nexts:
                # Dead end — we've reached an unterminated branch tip
                break

            # If the next pixel is a seed (endpoint/junction), stop there
            # and include it in the branch (so the branch spans from seed
            # to seed for complete angular coverage)
            next_seeds = [p for p in nexts if p in seed_set]
            if next_seeds:
                branch.append(next_seeds[0])
                break

            # Otherwise, take the single unvisited continuation pixel.
            # If there are multiple (i.e. we've hit another junction that
            # was not labelled as a seed), take the one with the lowest
            # valence to stay on the main branch.
            nxt     = nexts[0]
            visited[nxt[1], nxt[0]] = True
            branch.append(nxt)
            current = nxt

        return branch

    # -- Phase 6: suppression near anchors -----------------------------------

    @staticmethod
    def _suppress_near_anchors(
        turns  : List[Point],
        anchors: List[Point],
        radius : int,
    ) -> List[Point]:
        """
        Remove any turn candidate whose Euclidean distance from ANY anchor
        point (endpoint or junction) is less than *radius* pixels.

        This cleans up the raster artefacts that always appear where a
        branch root joins a junction cluster: the local direction changes
        abruptly at these spots even on straight roads, causing false bends.
        """
        if not turns or not anchors:
            return turns

        r2 = radius * radius   # compare squared distances to avoid sqrt
        kept: List[Point] = []
        for t in turns:
            tx, ty = t
            near_anchor = False
            for (ax, ay) in anchors:
                dx, dy = tx - ax, ty - ay
                if dx * dx + dy * dy < r2:
                    near_anchor = True
                    break
            if not near_anchor:
                kept.append(t)
        return kept

    # -- Phase 7: global node deduplication ----------------------------------

    def _global_merge(
        self,
        endpoints : List[Point],
        junctions : List[Point],
        turns     : List[Point],
    ) -> Tuple[List[Point], List[Point], List[Point]]:
        """
        Final deduplication pass.

        Rules (in priority order, high → low):
          1. Junctions have highest priority (they represent real topology).
          2. Endpoints have second priority.
          3. Turns have lowest priority.

        For every turn within merge_radius of a junction or endpoint, the
        turn is removed (the higher-priority node already covers it).

        For pairs within the SAME category, the one with the lexicographically
        smaller (x, y) is kept (arbitrary but deterministic).

        Parameters
        ----------
        endpoints : List[Point]
        junctions : List[Point]
        turns     : List[Point]

        Returns
        -------
        Tuple[List[Point], List[Point], List[Point]]
            Deduplicated (endpoints, junctions, turns).
        """
        r = self.merge_radius

        # Deduplicate within junctions
        junctions = _deduplicate(junctions, r)

        # Deduplicate within endpoints
        endpoints = _deduplicate(endpoints, r)

        # Remove endpoints that landed on a junction
        endpoints = [p for p in endpoints if not _any_within(p, junctions, r)]

        # Remove turns that landed on a junction or endpoint
        anchors = junctions + endpoints
        turns   = _deduplicate(turns, r)
        turns   = [t for t in turns if not _any_within(t, anchors, r)]

        return endpoints, junctions, turns


# ===========================================================================
# Module-level geometry utilities  (no class state needed)
# ===========================================================================

def _euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two (x,y) points."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _any_within(p: Point, others: List[Point], radius: int) -> bool:
    """Return True if *p* is within *radius* pixels of any point in *others*."""
    r2 = radius * radius
    px, py = p
    for (ox, oy) in others:
        dx, dy = px - ox, py - oy
        if dx * dx + dy * dy < r2:
            return True
    return False


def _deduplicate(points: List[Point], radius: int) -> List[Point]:
    """
    Remove near-duplicate points within a list.

    For every pair within *radius* pixels, keep only the first encountered
    (after sorting by x then y for deterministic output).

    This is O(n²) on the number of nodes, but node counts are always small
    (tens to low hundreds), so this is not a performance concern.
    """
    if not points:
        return []
    pts    = sorted(set(points))   # sort + dedupe exact duplicates
    kept   : List[Point] = []
    r2     = radius * radius
    removed= [False] * len(pts)

    for i, p in enumerate(pts):
        if removed[i]:
            continue
        kept.append(p)
        px, py = p
        for j in range(i + 1, len(pts)):
            if removed[j]:
                continue
            qx, qy = pts[j]
            dx, dy = px - qx, py - qy
            if dx * dx + dy * dy < r2:
                removed[j] = True

    return kept


def _turning_angle_deg(
    p_prev: Point,
    p_curr: Point,
    p_next: Point,
) -> float:
    """
    Compute the *turning angle* at *p_curr* in degrees.

    The turning angle is the angular deviation from straight travel:
      0°   → perfectly straight
      90°  → right-angle turn
      180° → U-turn

    Computed as:
      v1 = p_curr − p_prev   (incoming vector)
      v2 = p_next − p_curr   (outgoing vector)
      cos θ = (v1 · v2) / (|v1| |v2|)
      turning_angle = 180° − θ   where θ = arccos(cos θ)

    Parameters
    ----------
    p_prev, p_curr, p_next : Point  (x, y) tuples

    Returns
    -------
    float
        Turning angle in [0, 180] degrees.
        Returns 0.0 if either vector is degenerate (zero length).
    """
    v1x = p_curr[0] - p_prev[0]
    v1y = p_curr[1] - p_prev[1]
    v2x = p_next[0] - p_curr[0]
    v2y = p_next[1] - p_curr[1]

    len1 = math.sqrt(v1x * v1x + v1y * v1y)
    len2 = math.sqrt(v2x * v2x + v2y * v2y)

    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0

    cos_theta = (v1x * v2x + v1y * v2y) / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))   # numerical clamp
    theta_deg = math.degrees(math.acos(cos_theta))

    # theta_deg is the angle BETWEEN the two vectors.
    # turning angle = 180 - theta (deviation from straight)
    return 180.0 - theta_deg


# ===========================================================================
# Douglas-Peucker polyline simplification
# ===========================================================================

def _douglas_peucker(branch: List[Point], epsilon: float) -> List[int]:
    """
    Ramer-Douglas-Peucker polyline simplification.

    Reduces the ordered list of pixels in *branch* to a subset of indices
    that preserves the overall shape within tolerance *epsilon* pixels.

    Parameters
    ----------
    branch  : List[Point]  ordered (x, y) pixel list
    epsilon : float        maximum perpendicular deviation tolerance in px

    Returns
    -------
    List[int]
        Sorted list of indices into *branch* that form the simplified
        polyline.  Always includes index 0 and index len(branch)-1.

    Algorithm
    ---------
    Iterative (stack-based) implementation to avoid Python recursion depth
    limits on very long branches (e.g. 10 000+ pixels on a 2048×2048 mask).

    For each segment [start, end], find the point with maximum perpendicular
    distance from the segment line.  If that distance > epsilon, split there
    and recurse on both halves; otherwise, discard all interior points.
    """
    n = len(branch)
    if n < 3:
        return list(range(n))

    # Work with a boolean keep array for O(1) marking
    keep = np.zeros(n, dtype=bool)
    keep[0]     = True
    keep[n - 1] = True

    # Iterative stack instead of recursion
    stack = [(0, n - 1)]

    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue

        # Find the index with maximum perpendicular distance from
        # the line segment  branch[start] → branch[end]
        max_dist  = 0.0
        max_index = start

        x1, y1 = branch[start]
        x2, y2 = branch[end]

        # Line vector
        lx, ly   = x2 - x1, y2 - y1
        line_len = math.sqrt(lx * lx + ly * ly)

        for i in range(start + 1, end):
            px, py = branch[i]

            if line_len < 1e-9:
                # Degenerate segment: distance = point distance from start
                dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            else:
                # Perpendicular distance from point to the infinite line
                # |cross product| / |line vector|
                cross = abs(lx * (y1 - py) - ly * (x1 - px))
                dist  = cross / line_len

            if dist > max_dist:
                max_dist  = dist
                max_index = i

        if max_dist > epsilon:
            keep[max_index] = True
            stack.append((start, max_index))
            stack.append((max_index, end))

    return [i for i in range(n) if keep[i]]