"""
candidate_matcher.py
====================
RouteTREE – Candidate Gap Matching Module
Module 3 of 5 in the Road Healing Pipeline

PURPOSE
-------
Takes the raw gap candidates from RoadGapDetector and the direction
estimates from OrientationAnalyzer, and combines them into a single
enriched comparison for every candidate pair.

This module answers THREE geometric questions for each candidate gap:

    1. DISTANCE  — How far apart are the two endpoints?
                   Shorter gaps are more likely real occlusions.

    2. DIRECTION — Are the two road directions similar?
                   A true interrupted road has both endpoints pointing
                   in nearly the same overall direction (same road axis).

    3. FACING    — Are the two endpoints actually pointing TOWARD
                   each other?
                   A true gap has the two endpoints facing each other
                   (angle difference ≈ 180°).
                   Two endpoints facing away from each other or in
                   perpendicular directions are NOT a gap.

It REJECTS candidates that fail hard geometric constraints:
    - Facing angle difference < facing_threshold   (not facing each other)
    - Direction difference > direction_threshold   (roads go different ways)

It SCORES passing candidates with individual component scores and a
combined similarity score in [0, 1], which feeds directly into
GapScoring (Module 4) for the final weighted confidence computation.

POSITION IN PIPELINE
--------------------
RoadGapDetector     → raw candidates  (Module 1)
OrientationAnalyzer → direction data  (Module 2)
>>> CandidateMatcher <<<              ← YOU ARE HERE
GapScoring          → confidence      (Module 4)
GraphHealer         → edge insertion  (Module 5)

OUTPUT FORMAT
-------------
Each matched candidate dict contains ALL original fields PLUS:

    "direction_a"         : DirectionResult   full result for start_node
    "direction_b"         : DirectionResult   full result for end_node
    "angle_a"             : float             road angle at start endpoint
    "angle_b"             : float             road angle at end endpoint
    "facing_angle"        : float             angular difference (0-180°)
                                              ~180° = facing each other
    "direction_difference": float             axis alignment (0-180°)
                                              ~0° = same road axis
    "distance_score"      : float             [0,1]  1=close
    "direction_score"     : float             [0,1]  1=same axis
    "facing_score"        : float             [0,1]  1=perfectly facing
    "similarity_score"    : float             [0,1]  combined score
    "rejected"            : bool              True if hard-rejected
    "reject_reason"       : str               human-readable reason

USAGE
-----
    from candidate_matcher import CandidateMatcher

    matcher = CandidateMatcher(
        max_distance          = 75,
        facing_threshold      = 120.0,
        direction_threshold   = 60.0,
    )

    matched = matcher.match_candidates(candidates, directions, skeleton)

DEPENDENCIES
------------
    Python   >= 3.8
    NumPy    >= 1.24
    OpenCV   >= 4.x
    NetworkX is NOT needed here — we work on geometry only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from orientation_analyzer import OrientationAnalyzer


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point     = Tuple[int, int]
Candidate = Dict[str, Any]


class CandidateMatcher:
    """
    Enriches and filters gap candidates using geometric direction analysis.

    Parameters
    ----------
    max_distance : float
        Maximum gap distance in pixels.  Candidates beyond this are
        hard-rejected regardless of direction.  Should match or exceed
        the search_radius used in RoadGapDetector.  Default: 75.

    facing_threshold : float
        Minimum facing angle (degrees) for a candidate to be accepted.
        The facing angle is the angular difference between the two
        endpoint directions — it should be close to 180° for a real gap.
        Candidates with facing_angle < facing_threshold are rejected.
        Default: 120°.  (Allows up to 60° of misalignment from perfect
        180°, accounting for curved road approaches.)

    direction_threshold : float
        Maximum road axis difference (degrees) allowed between the two
        endpoints.  Two endpoints on the SAME road axis should have a
        direction difference close to 0° (or 180°, since direction is
        undirected for a road axis).
        Default: 60°.

    walk_length : int
        Walk length passed to OrientationAnalyzer when directions are
        not pre-computed.  Default: 20.
    """

    def __init__(
        self,
        max_distance        : float = 75.0,
        facing_threshold    : float = 120.0,
        direction_threshold : float = 60.0,
        walk_length         : int   = 20,
    ) -> None:
        self.max_distance        = float(max_distance)
        self.facing_threshold    = float(facing_threshold)
        self.direction_threshold = float(direction_threshold)
        self._analyzer           = OrientationAnalyzer(walk_length=walk_length)

    # -----------------------------------------------------------------------
    # Primary public method
    # -----------------------------------------------------------------------

    def match_candidates(
        self,
        candidates : List[Candidate],
        directions : Dict[Point, Any],
        skeleton   : np.ndarray,
    ) -> List[Candidate]:
        """
        Enrich every candidate with direction analysis and similarity scores.

        For each candidate:
            1. Look up (or compute) direction at start and end endpoints.
            2. Compute facing_angle and direction_difference.
            3. Apply hard rejection filters.
            4. If not rejected, compute component scores and similarity.
            5. Update the candidate dict in place (adds new keys).

        Parameters
        ----------
        candidates : List[Candidate]
            Output from RoadGapDetector.find_gaps().
        directions : Dict[Point, DirectionResult]
            Output from OrientationAnalyzer.analyze_all().
            If a point is missing, direction is computed on the fly.
        skeleton   : np.ndarray
            Binary skeleton (needed for on-the-fly direction estimation).

        Returns
        -------
        List[Candidate]
            Same list, each dict enriched with matching fields.
            Rejected candidates are included but flagged with
            "rejected": True and "reject_reason": "...".
            Sorted by similarity_score descending (best matches first).
        """
        for candidate in candidates:
            self._enrich_candidate(candidate, directions, skeleton)

        # Sort: accepted first (by similarity desc), then rejected last
        candidates.sort(
            key=lambda c: (c["rejected"], -c.get("similarity_score", 0.0))
        )

        return candidates

    # -----------------------------------------------------------------------
    # Named public scoring functions
    # -----------------------------------------------------------------------

    def compare_distance(self, distance: float) -> float:
        """
        Score the Euclidean distance between two gap endpoints.

        Scoring rationale
        -----------------
        Very short gaps (< 10px) score highest — they are most likely
        real occlusions since the road skeleton is nearly intact.

        Longer gaps score lower, falling to 0.0 at max_distance.

        Uses an exponential decay so short gaps are strongly preferred:
            score = exp(-distance / (max_distance / 3))

        Parameters
        ----------
        distance : float   gap distance in pixels

        Returns
        -------
        float in [0, 1]    1.0 = very close,  0.0 = at or beyond max_distance
        """
        if distance <= 0.0:
            return 1.0
        if distance >= self.max_distance:
            return 0.0
        # Exponential decay: decays to ~0.05 at max_distance
        return math.exp(-distance / (self.max_distance / 3.0))

    @staticmethod
    def compare_direction(direction_difference: float) -> float:
        """
        Score how well aligned the two road AXES are.

        The direction difference is the angular difference between the
        two endpoint direction vectors treated as undirected axes
        (so 0° and 180° both mean "same axis").

        Scoring
        -------
        0°  difference → score 1.0  (perfectly same axis)
        60° difference → score ~0.5
        90° difference → score 0.0  (perpendicular roads — not a gap)

        Uses a cosine-based mapping for smooth falloff.

        Parameters
        ----------
        direction_difference : float   [0, 90] degrees (axis difference)

        Returns
        -------
        float in [0, 1]
        """
        # Clamp to valid range
        diff = max(0.0, min(90.0, direction_difference))
        # Cosine maps 0→1.0, 90→0.0 with smooth curve
        return math.cos(math.radians(diff))

    @staticmethod
    def compare_angle(facing_angle: float) -> float:
        """
        Score how directly the two endpoints are facing each other.

        The facing angle is the angular difference between the two
        endpoint direction vectors (DIRECTED, not axis).
        For a real gap, this should be close to 180°.

        Scoring
        -------
        180° difference → score 1.0  (perfectly facing)
        120° difference → score ~0.5 (somewhat facing — acceptable)
        90°  difference → score 0.0  (perpendicular — not facing)
        <90° difference → negative   (facing same direction — definitely not a gap)

        Parameters
        ----------
        facing_angle : float   [0, 180] degrees

        Returns
        -------
        float in [0, 1]   (clamped to 0.0 for facing_angle < 90°)
        """
        if facing_angle < 90.0:
            return 0.0
        # Linear mapping: 90°→0.0, 180°→1.0
        return (facing_angle - 90.0) / 90.0

    # -----------------------------------------------------------------------
    # Individual comparison helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _axis_difference(angle_a: float, angle_b: float) -> float:
        """
        Compute the difference between two road AXES (undirected angles).

        Unlike facing_angle (which cares about direction), axis_difference
        treats a road at 10° and a road at 190° as the SAME axis (both
        are the same road, just measured from opposite ends).

        Returns a value in [0, 90] where:
            0° = same axis (parallel roads or same road)
            90° = perpendicular roads

        Parameters
        ----------
        angle_a, angle_b : float  angles in degrees [0, 360)

        Returns
        -------
        float in [0, 90]
        """
        # Fold both angles into [0, 180) to make them undirected
        a = angle_a % 180.0
        b = angle_b % 180.0
        diff = abs(a - b)
        # Fold again into [0, 90]
        if diff > 90.0:
            diff = 180.0 - diff
        return diff

    # -----------------------------------------------------------------------
    # Core enrichment
    # -----------------------------------------------------------------------

    def _enrich_candidate(
        self,
        candidate  : Candidate,
        directions : Dict[Point, Any],
        skeleton   : np.ndarray,
    ) -> None:
        """
        Enrich a single candidate dict in-place.

        Adds all matching fields to the candidate dict.
        """
        pa = candidate["start_pos"]
        pb = candidate["end_pos"]

        # Step 1: get or compute direction at each endpoint
        dir_a = self._get_direction(pa, directions, skeleton)
        dir_b = self._get_direction(pb, directions, skeleton)

        candidate["direction_a"] = dir_a
        candidate["direction_b"] = dir_b

        # Step 2: extract angles
        angle_a = dir_a["angle_deg"] if dir_a["valid"] else None
        angle_b = dir_b["angle_deg"] if dir_b["valid"] else None

        candidate["angle_a"] = angle_a
        candidate["angle_b"] = angle_b

        # Step 3: handle case where direction estimation failed
        if angle_a is None or angle_b is None:
            self._mark_rejected(candidate, "Direction estimation failed at one or both endpoints")
            return

        # Step 4: compute facing angle (directed — should be ~180° for real gap)
        facing_angle = OrientationAnalyzer.angle_difference(angle_a, angle_b)
        candidate["facing_angle"] = round(facing_angle, 2)

        # Step 5: compute road axis difference (undirected — should be ~0° for same road)
        axis_diff = self._axis_difference(angle_a, angle_b)
        candidate["direction_difference"] = round(axis_diff, 2)

        # Step 6: hard rejection filters
        if facing_angle < self.facing_threshold:
            self._mark_rejected(
                candidate,
                f"Facing angle {facing_angle:.1f}° < threshold {self.facing_threshold:.1f}°"
                f" — endpoints not facing each other"
            )
            return

        if axis_diff > self.direction_threshold:
            self._mark_rejected(
                candidate,
                f"Direction difference {axis_diff:.1f}° > threshold {self.direction_threshold:.1f}°"
                f" — roads have different orientations"
            )
            return

        if candidate["distance"] > self.max_distance:
            self._mark_rejected(
                candidate,
                f"Distance {candidate['distance']:.1f}px > max {self.max_distance:.1f}px"
            )
            return

        # Step 7: compute component scores
        dist_score  = self.compare_distance(candidate["distance"])
        dir_score   = self.compare_direction(axis_diff)
        face_score  = self.compare_angle(facing_angle)

        candidate["distance_score"]  = round(dist_score,  4)
        candidate["direction_score"] = round(dir_score,   4)
        candidate["facing_score"]    = round(face_score,  4)

        # Step 8: combined similarity score (equal weights here;
        # GapScoring applies configurable weights in Module 4)
        similarity = (dist_score + dir_score + face_score) / 3.0
        candidate["similarity_score"] = round(similarity, 4)
        candidate["rejected"]         = False
        candidate["reject_reason"]    = ""

    def _get_direction(
        self,
        point      : Point,
        directions : Dict[Point, Any],
        skeleton   : np.ndarray,
    ) -> Any:
        """
        Look up a pre-computed direction result, or compute it on the fly
        if the point is not in the directions dict.

        This makes the module robust to partial direction caches — e.g.
        if directions was computed only for degree-1 endpoints but the
        candidate includes a turn node or junction that wasn't analyzed.
        """
        if point in directions:
            return directions[point]
        # Not pre-computed — estimate now
        return self._analyzer.estimate_direction(skeleton, point)

    @staticmethod
    def _mark_rejected(candidate: Candidate, reason: str) -> None:
        """Mark a candidate as rejected and zero out its scores."""
        candidate["facing_angle"]        = candidate.get("facing_angle", -1.0)
        candidate["direction_difference"] = candidate.get("direction_difference", -1.0)
        candidate["distance_score"]       = 0.0
        candidate["direction_score"]      = 0.0
        candidate["facing_score"]         = 0.0
        candidate["similarity_score"]     = 0.0
        candidate["rejected"]             = True
        candidate["reject_reason"]        = reason

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def visualize(
        self,
        image      : np.ndarray,
        candidates : List[Candidate],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize matched and rejected candidates with color coding.

        Color legend
        ------------
            Green  dashed  : accepted candidate (similarity >= 0.6)
            Yellow dashed  : accepted candidate (similarity < 0.6)
            Gray   dashed  : rejected candidate
            Orange arrows  : direction vectors at endpoints
            White  text    : similarity score label

        Parameters
        ----------
        image       : np.ndarray   BGR image to annotate
        candidates  : List[Candidate]  output from match_candidates()
        output_path : str          optional save path

        Returns
        -------
        np.ndarray  annotated BGR image
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for c in candidates:
            p1 = c["start_pos"]
            p2 = c["end_pos"]

            if c["rejected"]:
                color     = (100, 100, 100)   # gray
                thickness = 1
            elif c["similarity_score"] >= 0.6:
                color     = (0, 255, 0)        # green
                thickness = 2
            else:
                color     = (0, 255, 255)      # yellow
                thickness = 1

            # Dashed gap line
            self._draw_dashed_line(canvas, p1, p2, color, thickness)

            # Direction arrows at endpoints
            for key in ("direction_a", "direction_b"):
                dr = c.get(key)
                if dr and dr.get("valid"):
                    ep = dr["endpoint"]
                    ux, uy = dr["unit_vector"]
                    tip = (
                        int(round(ep[0] + ux * 20)),
                        int(round(ep[1] + uy * 20)),
                    )
                    cv2.arrowedLine(canvas, ep, tip, (0, 165, 255), 1,
                                    tipLength=0.4, line_type=cv2.LINE_AA)

            # Score / reject label at midpoint
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            if c["rejected"]:
                label = "X"
            else:
                label = f"{c['similarity_score']:.2f}"

            cv2.putText(canvas, label, (mid[0] + 3, mid[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            # Endpoint markers
            cv2.circle(canvas, p1, 4, color, -1)
            cv2.circle(canvas, p2, 4, color, -1)

        # Legend
        self._draw_legend(canvas)

        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"  Visualization saved → {output_path}")

        return canvas

    @staticmethod
    def _draw_dashed_line(
        image    : np.ndarray,
        p1       : Point,
        p2       : Point,
        color    : Tuple,
        thickness: int = 1,
        dash_len : int = 8,
    ) -> None:
        """Draw a dashed line between p1 and p2."""
        x1, y1 = p1
        x2, y2 = p2
        total   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if total < 1:
            return
        ux = (x2 - x1) / total
        uy = (y2 - y1) / total
        dist = 0.0
        draw = True
        while dist < total:
            seg_end = min(dist + dash_len, total)
            if draw:
                sx1 = int(x1 + ux * dist);   sy1 = int(y1 + uy * dist)
                sx2 = int(x1 + ux * seg_end); sy2 = int(y1 + uy * seg_end)
                cv2.line(image, (sx1, sy1), (sx2, sy2), color, thickness)
            dist += dash_len
            draw  = not draw

    @staticmethod
    def _draw_legend(canvas: np.ndarray) -> None:
        items = [
            ((0, 255, 0),     "Accepted (high confidence)"),
            ((0, 255, 255),   "Accepted (low confidence)"),
            ((100, 100, 100), "Rejected"),
            ((0, 165, 255),   "Road direction"),
        ]
        x, y = 8, 16
        for color, label in items:
            cv2.rectangle(canvas, (x, y - 8), (x + 12, y + 2), color, -1)
            cv2.putText(canvas, label, (x + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 16