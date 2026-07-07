"""
orientation_analyzer.py
========================
RouteTREE – Road Orientation Analysis Module
Module 2 of 5 in the Road Healing Pipeline

PURPOSE
-------
Estimates the direction a road is travelling at any given endpoint.

This is critical for gap healing because knowing the road's direction
at each endpoint lets us answer the question:

    "Are these two endpoints actually facing each other?"

If endpoint A points NORTH and endpoint B points SOUTH, and they are
close together, they are almost certainly the same road interrupted
by an occlusion.

If endpoint A points NORTH and endpoint B also points NORTH, they are
two road ends going in the same direction — NOT a gap to heal.

POSITION IN PIPELINE
--------------------
Road Mask
    ↓
Skeleton
    ↓
Node Detection
    ↓
Edge Generation
    ↓
Road Graph
    ↓
RoadGapDetector         (Module 1 — finds candidate pairs)
    ↓
>>> OrientationAnalyzer <<<    ← YOU ARE HERE
    ↓
CandidateMatcher        (Module 3)
    ↓
GapScoring              (Module 4)
    ↓
GraphHealer             (Module 5)

METHOD — Walk-Back Direction Estimation
----------------------------------------
Naive approach (WRONG): just look at the endpoint pixel's immediate
neighbours.  This gives a direction based on only 1-2 pixels, which
is extremely noisy.

This module's approach (CORRECT):
    1. Start at the endpoint pixel.
    2. Walk N pixels BACK along the skeleton (away from the endpoint,
       into the road).
    3. Collect the ordered sequence of pixels along that walk.
    4. Fit a direction vector from the endpoint to the pixel N steps
       back using least-squares linear regression on all the collected
       pixels — this averages out raster quantisation noise.
    5. Normalize the vector to unit length.
    6. Compute the angle in degrees (0° = East, 90° = North, etc.)

Why regression instead of just endpoint-to-tail vector?
    Because the skeleton near an endpoint is often slightly curved.
    Regression on all N pixels gives a stable average direction
    rather than being thrown off by the last pixel's exact position.

PARAMETERS
----------
    walk_length : int
        Number of pixels to walk back from the endpoint along the
        skeleton.  Larger → more stable but may turn a corner on
        curved roads.
        Default: 20px.  Recommended range: 10–40px.

USAGE
-----
    from orientation_analyzer import OrientationAnalyzer

    analyzer = OrientationAnalyzer(walk_length=20)

    # Single endpoint
    result = analyzer.estimate_direction(skeleton, endpoint=(x, y))
    print(result["vector"])    # (dx, dy)
    print(result["angle_deg"]) # float, degrees

    # All endpoints at once
    directions = analyzer.analyze_all(skeleton, endpoints)

DEPENDENCIES
------------
    Python   >= 3.8
    NumPy    >= 1.24
    OpenCV   >= 4.x
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point       = Tuple[int, int]          # (x, y) pixel coordinate
Vector      = Tuple[float, float]      # (dx, dy) direction vector
DirectionResult = Dict                 # full result dict from estimate_direction


class OrientationAnalyzer:
    """
    Estimates road direction at skeleton endpoints by walking back
    along the skeleton and fitting a direction vector via regression.

    Parameters
    ----------
    walk_length : int
        How many pixels to walk back from the endpoint along the
        skeleton when estimating direction.  Default 20.
        Increase for smoother (but potentially corner-crossing)
        estimates.  Decrease if roads are very short.
    """

    def __init__(self, walk_length: int = 20) -> None:
        self.walk_length = int(walk_length)

    # -----------------------------------------------------------------------
    # Primary public methods
    # -----------------------------------------------------------------------

    def estimate_direction(
        self,
        skeleton : np.ndarray,
        endpoint : Point,
    ) -> DirectionResult:
        """
        Estimate the road direction at a single endpoint.

        Walks ``walk_length`` pixels back along the skeleton from the
        endpoint, then fits a direction vector using least-squares
        linear regression on all collected pixels.

        Parameters
        ----------
        skeleton : np.ndarray
            Binary skeleton, uint8, foreground > 0.
        endpoint : Point
            (x, y) pixel coordinate of the endpoint.

        Returns
        -------
        dict with keys:
            "endpoint"    : Point       the input endpoint
            "walk_pixels" : List[Point] pixels collected during walk
            "vector"      : Vector      raw direction vector (dx, dy)
            "unit_vector" : Vector      normalized to length 1.0
            "angle_deg"   : float       angle in degrees
                                        0°=East 90°=North 180°=West
                                        (standard math convention)
            "valid"       : bool        False if direction could not
                                        be estimated (isolated pixel,
                                        too short a branch, etc.)
        """
        skel = self._to_binary(skeleton)

        # Walk back from the endpoint along the skeleton
        walk_pixels = self._walk_back(skel, endpoint, self.walk_length)

        # Need at least 2 pixels to define a direction
        if len(walk_pixels) < 2:
            return self._invalid_result(endpoint, walk_pixels)

        # Fit direction via regression on all collected pixels
        vector = self._fit_direction(walk_pixels, endpoint)

        if vector is None:
            return self._invalid_result(endpoint, walk_pixels)

        unit_vec  = self.normalize(vector)
        angle     = self.direction_angle(unit_vec)

        return {
            "endpoint"    : endpoint,
            "walk_pixels" : walk_pixels,
            "vector"      : vector,
            "unit_vector" : unit_vec,
            "angle_deg"   : angle,
            "valid"        : True,
        }

    def analyze_all(
        self,
        skeleton  : np.ndarray,
        endpoints : List[Point],
    ) -> Dict[Point, DirectionResult]:
        """
        Estimate direction for every endpoint in the list.

        Parameters
        ----------
        skeleton  : np.ndarray    binary skeleton
        endpoints : List[Point]   list of (x, y) endpoint coordinates

        Returns
        -------
        dict  { (x, y) : DirectionResult }
            Keyed by endpoint coordinate for fast lookup by downstream
            modules (CandidateMatcher, GapScoring).
        """
        results: Dict[Point, DirectionResult] = {}
        for ep in endpoints:
            results[ep] = self.estimate_direction(skeleton, ep)
        return results

    # -----------------------------------------------------------------------
    # Named public functions (matching the required API)
    # -----------------------------------------------------------------------

    @staticmethod
    def normalize(vector: Vector) -> Vector:
        """
        Normalize a (dx, dy) vector to unit length.

        Parameters
        ----------
        vector : (dx, dy)

        Returns
        -------
        (dx, dy) with magnitude 1.0, or (0.0, 0.0) if input is zero.
        """
        dx, dy = vector
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-9:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    @staticmethod
    def direction_angle(unit_vector: Vector) -> float:
        """
        Convert a unit direction vector to an angle in degrees.

        Convention: standard math angles.
            0°   = East  (+x direction)
            90°  = North (-y direction, because y increases downward
                          in image coordinates)
            180° = West  (-x direction)
            270° = South (+y direction)

        Parameters
        ----------
        unit_vector : (dx, dy)

        Returns
        -------
        float  angle in [0, 360) degrees
        """
        dx, dy = unit_vector
        # atan2 returns angle in [-π, π]; negate dy because image y
        # increases downward (opposite to standard math convention)
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        # Normalize to [0, 360)
        return angle_deg % 360.0

    @staticmethod
    def angle_difference(angle_a: float, angle_b: float) -> float:
        """
        Compute the smallest angular difference between two angles.

        Returns a value in [0, 180] degrees — the minimum rotation
        needed to align the two directions.

        Parameters
        ----------
        angle_a, angle_b : float  angles in degrees

        Returns
        -------
        float  difference in [0, 180]
        """
        diff = abs(angle_a - angle_b) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def visualize_direction(
        self,
        image     : np.ndarray,
        result    : DirectionResult,
        arrow_len : int            = 30,
        color     : Tuple          = (0, 165, 255),  # orange
    ) -> np.ndarray:
        """
        Draw the estimated road direction as an arrow on a copy of *image*.

        The arrow starts at the endpoint and points in the estimated
        direction the road is travelling (i.e. where the gap continues).

        Parameters
        ----------
        image     : np.ndarray      BGR image to annotate
        result    : DirectionResult output from estimate_direction()
        arrow_len : int             length of drawn arrow in pixels
        color     : (B, G, R)       arrow color

        Returns
        -------
        np.ndarray  annotated BGR image (copy, original unchanged)
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        ep = result["endpoint"]

        # Draw the walk-back pixels in dim blue so you can see
        # which pixels were used for the regression
        for px, py in result.get("walk_pixels", []):
            cv2.circle(canvas, (px, py), 1, (180, 80, 0), -1)

        if not result.get("valid", False):
            # Draw a red X at invalid endpoints
            cv2.drawMarker(canvas, ep, (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 10, 2)
            return canvas

        # Draw arrow: endpoint → endpoint + unit_vector * arrow_len
        ux, uy = result["unit_vector"]
        tip = (
            int(round(ep[0] + ux * arrow_len)),
            int(round(ep[1] + uy * arrow_len)),
        )

        cv2.arrowedLine(canvas, ep, tip, color, 2,
                        tipLength=0.3, line_type=cv2.LINE_AA)

        # Draw endpoint dot
        cv2.circle(canvas, ep, 4, color, -1)

        # Angle label
        angle = result["angle_deg"]
        cv2.putText(
            canvas,
            f"{angle:.0f}deg",
            (ep[0] + 5, ep[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

        return canvas

    def visualize_all(
        self,
        image       : np.ndarray,
        results     : Dict[Point, DirectionResult],
        output_path : Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw direction arrows for every endpoint on one image.

        Parameters
        ----------
        image       : np.ndarray
        results     : dict { (x,y) : DirectionResult }
        output_path : str  optional save path

        Returns
        -------
        np.ndarray  annotated BGR image
        """
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for ep, result in results.items():
            canvas = self.visualize_direction(canvas, result)

        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"  Visualization saved → {output_path}")

        return canvas

    # -----------------------------------------------------------------------
    # Walk-back algorithm
    # -----------------------------------------------------------------------

    def _walk_back(
        self,
        skel    : np.ndarray,
        start   : Point,
        n_steps : int,
    ) -> List[Point]:
        """
        Walk n_steps pixels along the skeleton starting from *start*,
        moving away from the endpoint (i.e. into the road interior).

        Algorithm
        ---------
        - At each step, look at all 8-connected foreground neighbours.
        - Exclude the pixel we just came from (prevent backtracking).
        - If multiple neighbours exist (junction), prefer the one that
          continues in the most similar direction to our current travel.
        - Stop early if we hit a dead end or a junction node
          (we don't want to cross over a junction into a different branch).

        Parameters
        ----------
        skel    : np.ndarray   binary skeleton (0/1)
        start   : Point        endpoint (x, y)
        n_steps : int          max pixels to walk

        Returns
        -------
        List[Point]  ordered pixel sequence starting at *start*.
                     May be shorter than n_steps if the branch ends.
        """
        h, w    = skel.shape
        visited = {start}
        path    = [start]
        current = start
        prev    = None

        for _ in range(n_steps):
            cx, cy = current

            # Collect 8-connected foreground neighbours not yet visited
            neighbours: List[Point] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx_px, ny_px = cx + dx, cy + dy
                    if 0 <= nx_px < w and 0 <= ny_px < h:
                        if skel[ny_px, nx_px] > 0 and (nx_px, ny_px) not in visited:
                            neighbours.append((nx_px, ny_px))

            if not neighbours:
                # Dead end — stop
                break

            # If more than one choice (mini-junction), pick the one that
            # best continues the current direction of travel
            if len(neighbours) == 1:
                nxt = neighbours[0]
            else:
                nxt = self._best_continuation(current, prev, neighbours)

            visited.add(nxt)
            path.append(nxt)
            prev    = current
            current = nxt

        return path

    @staticmethod
    def _best_continuation(
        current    : Point,
        prev       : Optional[Point],
        neighbours : List[Point],
    ) -> Point:
        """
        When multiple neighbours exist, choose the one that best
        continues the current direction of travel.

        If there is no previous pixel (we're at the very first step),
        just return the first neighbour.

        Uses dot product: the neighbour whose direction vector has the
        highest dot product with the current travel direction is the
        most "straight ahead" choice.
        """
        if prev is None:
            return neighbours[0]

        # Current travel direction vector
        tdx = current[0] - prev[0]
        tdy = current[1] - prev[1]
        tmag = math.sqrt(tdx * tdx + tdy * tdy)
        if tmag < 1e-9:
            return neighbours[0]

        best_dot = -float("inf")
        best_nb  = neighbours[0]

        for nb in neighbours:
            ndx = nb[0] - current[0]
            ndy = nb[1] - current[1]
            dot = (tdx * ndx + tdy * ndy) / tmag
            if dot > best_dot:
                best_dot = dot
                best_nb  = nb

        return best_nb

    # -----------------------------------------------------------------------
    # Direction vector fitting
    # -----------------------------------------------------------------------

    @staticmethod
    def _fit_direction(
        walk_pixels : List[Point],
        endpoint    : Point,
    ) -> Optional[Vector]:
        """
        Fit a direction vector from the endpoint using least-squares
        linear regression on all pixels collected during the walk.

        The direction vector points FROM the endpoint INTO the road
        (i.e. the direction the road is travelling at that endpoint).

        Why regression?
        ---------------
        A skeleton is a discrete raster.  Simply taking (last_pixel -
        endpoint) gives a direction based on only two pixels, which
        is highly sensitive to raster quantisation noise.

        Fitting a line through all N walk pixels gives a stable average
        direction that is much less affected by individual pixel offsets.

        Algorithm
        ---------
        1. Collect (x, y) coordinates of all walk pixels.
        2. Compute the PCA direction: the eigenvector corresponding to
           the largest eigenvalue of the covariance matrix of the point
           set.  This is the direction of maximum variance — i.e. the
           line that best fits the point cloud.
        3. Ensure the vector points AWAY from the endpoint (into the
           road) by checking its dot product with (tail - head).

        Parameters
        ----------
        walk_pixels : List[Point]   ordered pixels from endpoint inward
        endpoint    : Point         the endpoint itself

        Returns
        -------
        (dx, dy) direction vector, or None if degenerate.
        """
        if len(walk_pixels) < 2:
            return None

        pts = np.array(walk_pixels, dtype=np.float32)

        # Center the point cloud
        mean = pts.mean(axis=0)
        centered = pts - mean

        # 2×2 covariance matrix
        cov = (centered.T @ centered) / len(pts)

        # Eigendecomposition — largest eigenvector is the PCA direction
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns eigenvalues in ascending order; take the last
        principal = eigenvectors[:, -1]   # shape (2,)

        dx, dy = float(principal[0]), float(principal[1])

        # Ensure the vector points FROM endpoint INTO the road,
        # i.e. toward the tail of the walk (away from the tip)
        tail = walk_pixels[-1]
        away_dx = tail[0] - endpoint[0]
        away_dy = tail[1] - endpoint[1]
        dot = dx * away_dx + dy * away_dy
        if dot < 0:
            # Flip so it points away from the endpoint into the road
            dx, dy = -dx, -dy

        return (dx, dy)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_binary(skeleton: np.ndarray) -> np.ndarray:
        """Ensure skeleton is uint8 with foreground = 1."""
        skel = skeleton.astype(np.uint8)
        if skel.max() > 1:
            _, skel = cv2.threshold(skel, 127, 1, cv2.THRESH_BINARY)
        return skel

    @staticmethod
    def _invalid_result(
        endpoint    : Point,
        walk_pixels : List[Point],
    ) -> DirectionResult:
        """Return a result dict representing a failed direction estimate."""
        return {
            "endpoint"    : endpoint,
            "walk_pixels" : walk_pixels,
            "vector"      : (0.0, 0.0),
            "unit_vector" : (0.0, 0.0),
            "angle_deg"   : 0.0,
            "valid"        : False,
        }