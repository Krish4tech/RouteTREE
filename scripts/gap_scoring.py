"""
gap_scoring.py
==============

RouteTREE Module 4: GapScoring

Purpose
-------
Takes the enriched candidate list produced by
`candidate_matcher.CandidateMatcher.match_candidates()` and converts the
three raw geometric similarity scores already attached to each candidate
(distance_score, direction_score, facing_score -- all in [0, 1]) into a
single final `confidence` value in [0, 1], plus a human readable
`confidence_grade` ("HIGH" / "MEDIUM" / "LOW" / "REJECTED") and a
`score_breakdown` dict that shows exactly how the final number was built.

This module does NOT re-derive geometry (distance/angles/direction) --
that work is already done upstream by RoadGapDetector, OrientationAnalyzer
and CandidateMatcher. GapScoring is purely the "decision layer" that turns
those already-computed similarity scores into a final confidence + grade.

Dependencies: Python standard library, OpenCV (cv2), NumPy only.
"""

import cv2
import numpy as np


class GapScoring:
    """
    Combines the three raw similarity scores attached to a gap candidate
    (distance_score, direction_score, facing_score) into one final
    confidence value, using a configurable weighting scheme and a
    configurable combination method (weighted_sum / geometric / harmonic).
    """

    # Combination methods supported by overall_confidence()
    VALID_COMBINATIONS = ("weighted_sum", "geometric", "harmonic")

    # Small epsilon used to avoid literal division-by-zero in the
    # harmonic mean branch. Zero-score handling itself is done explicitly
    # (any raw score of 0 forces confidence to 0 for geometric/harmonic),
    # this epsilon only guards against float underflow noise.
    _EPS = 1e-9

    def __init__(self,
                 weight_distance=0.35,
                 weight_direction=0.30,
                 weight_facing=0.35,
                 combination="weighted_sum",
                 high_threshold=0.65,
                 low_threshold=0.40):

        if combination not in self.VALID_COMBINATIONS:
            raise ValueError(
                f"combination must be one of {self.VALID_COMBINATIONS}, "
                f"got '{combination}'"
            )

        if high_threshold <= low_threshold:
            raise ValueError(
                "high_threshold must be strictly greater than low_threshold "
                f"(got high={high_threshold}, low={low_threshold})"
            )

        # --- Auto-normalize weights so they always sum to exactly 1.0 ---
        # This lets the caller pass "relative importance" numbers (e.g.
        # 0.35, 0.30, 0.35, or even non-normalized values like 7, 6, 7)
        # without having to do the math themselves.
        raw_sum = weight_distance + weight_direction + weight_facing
        if raw_sum <= 0:
            raise ValueError("Sum of weights must be > 0")

        self.weight_distance = weight_distance / raw_sum
        self.weight_direction = weight_direction / raw_sum
        self.weight_facing = weight_facing / raw_sum

        self.combination = combination
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    # ------------------------------------------------------------------
    # Per-component weighted score helpers
    # ------------------------------------------------------------------
    # These return the WEIGHTED contribution of each raw component. They
    # are used directly by the "weighted_sum" combination method, and are
    # also reported (for transparency) inside score_breakdown regardless
    # of which combination method is active.

    def distance_score(self, raw):
        """Apply the configured distance weight to a raw distance score."""
        raw = self._clamp01(raw)
        return raw * self.weight_distance

    def direction_score(self, raw):
        """Apply the configured direction weight to a raw direction score."""
        raw = self._clamp01(raw)
        return raw * self.weight_direction

    def facing_score(self, raw):
        """Apply the configured facing weight to a raw facing score."""
        raw = self._clamp01(raw)
        return raw * self.weight_facing

    @staticmethod
    def _clamp01(value):
        """Clamp a numeric value into the closed interval [0, 1]."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    # ------------------------------------------------------------------
    # Combination logic
    # ------------------------------------------------------------------

    def overall_confidence(self, dist, dir, face):
        """
        Combine the three RAW (unweighted, [0,1]) component scores into a
        single final confidence value in [0, 1], using self.combination.

        Parameters
        ----------
        dist : float  raw distance_score  (from CandidateMatcher)
        dir  : float  raw direction_score (from CandidateMatcher)
        face : float  raw facing_score    (from CandidateMatcher)

        Returns
        -------
        float in [0, 1]
        """
        dist = self._clamp01(dist)
        dir = self._clamp01(dir)
        face = self._clamp01(face)

        if self.combination == "weighted_sum":
            # Standard weighted average. Weights already sum to 1.0, so
            # this is naturally bounded in [0, 1].
            confidence = (
                self.weight_distance * dist
                + self.weight_direction * dir
                + self.weight_facing * face
            )

        elif self.combination == "geometric":
            # Weighted geometric mean: product of (score ^ weight).
            # If ANY component is exactly 0, the whole geometric mean
            # collapses to 0 -- a single catastrophic disagreement
            # (e.g. roads not facing each other at all) should be able
            # to veto an otherwise-good candidate.
            if dist <= 0.0 or dir <= 0.0 or face <= 0.0:
                confidence = 0.0
            else:
                confidence = (
                    (dist ** self.weight_distance)
                    * (dir ** self.weight_direction)
                    * (face ** self.weight_facing)
                )

        else:  # "harmonic"
            # Weighted harmonic mean: 1 / (sum of weight_i / score_i).
            # The harmonic mean is dominated by the smallest input, so it
            # applies the strongest penalty of the three methods when any
            # single component score is weak, while still being 0 outright
            # if any component is exactly 0 (division by zero -> reject).
            if dist <= 0.0 or dir <= 0.0 or face <= 0.0:
                confidence = 0.0
            else:
                denominator = (
                    self.weight_distance / max(dist, self._EPS)
                    + self.weight_direction / max(dir, self._EPS)
                    + self.weight_facing / max(face, self._EPS)
                )
                confidence = 1.0 / denominator if denominator > 0 else 0.0

        # Final safety clamp -- floating point rounding could nudge
        # values a hair outside [0, 1] in edge cases.
        return self._clamp01(confidence)

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def grade(self, confidence):
        """
        Map a confidence value in [0, 1] to a qualitative grade.
        Note: this method only ever returns "HIGH" / "MEDIUM" / "LOW".
        The "REJECTED" grade is applied separately in score_candidates()
        for candidates that CandidateMatcher already flagged as
        geometrically invalid (rejected=True), regardless of what their
        numeric confidence would otherwise have been.
        """
        confidence = self._clamp01(confidence)
        if confidence >= self.high_threshold:
            return "HIGH"
        elif confidence >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_candidates(self, candidates):
        """
        Main method. Takes the enriched candidate list produced by
        CandidateMatcher.match_candidates() and adds, to each candidate
        dict (in place):

            confidence       : float in [0, 1]
            confidence_grade : "HIGH" / "MEDIUM" / "LOW" / "REJECTED"
            score_breakdown  : dict of per-component weighted contributions

        Returns the same list of dicts, re-sorted so the highest
        confidence candidates come first (REJECTED candidates sort last,
        since their confidence is forced to 0.0).

        Parameters
        ----------
        candidates : list[dict]
            Each dict is expected to already contain (from CandidateMatcher):
                distance_score, direction_score, facing_score : float [0,1]
                rejected : bool
                reject_reason : str or None

        Returns
        -------
        list[dict] sorted by confidence descending.
        """
        for candidate in candidates:
            raw_dist = candidate.get("distance_score", 0.0)
            raw_dir = candidate.get("direction_score", 0.0)
            raw_face = candidate.get("facing_score", 0.0)

            weighted_dist = self.distance_score(raw_dist)
            weighted_dir = self.direction_score(raw_dir)
            weighted_face = self.facing_score(raw_face)

            is_rejected = bool(candidate.get("rejected", False))

            if is_rejected:
                # A candidate that CandidateMatcher already hard-rejected
                # (facing_angle too small, or axis_diff too large) can
                # never be healed, no matter what its raw scores say.
                # We still compute what its confidence *would* have been
                # for transparency in the breakdown/summary, but the
                # confidence itself is forced to 0 and the grade is
                # forced to "REJECTED".
                would_be_confidence = self.overall_confidence(
                    raw_dist, raw_dir, raw_face
                )
                confidence = 0.0
                grade = "REJECTED"
            else:
                confidence = self.overall_confidence(raw_dist, raw_dir, raw_face)
                would_be_confidence = confidence
                grade = self.grade(confidence)

            candidate["confidence"] = confidence
            candidate["confidence_grade"] = grade
            candidate["score_breakdown"] = {
                "distance_raw": raw_dist,
                "distance_weighted": weighted_dist,
                "direction_raw": raw_dir,
                "direction_weighted": weighted_dir,
                "facing_raw": raw_face,
                "facing_weighted": weighted_face,
                "unweighted_sum_check": weighted_dist + weighted_dir + weighted_face,
                "combination_method": self.combination,
                "would_be_confidence": would_be_confidence,
                "weights": {
                    "distance": self.weight_distance,
                    "direction": self.weight_direction,
                    "facing": self.weight_facing,
                },
                "rejected_upstream": is_rejected,
                "reject_reason": candidate.get("reject_reason", None),
            }

        # Sort highest confidence first. Rejected candidates (confidence
        # forced to 0.0) naturally fall to the bottom. Python's sort is
        # stable, so ties preserve their original relative order.
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(candidates):
        """
        Print a formatted table of every scored candidate: endpoints,
        raw component scores, final confidence and grade.

        Expects candidates to already have been processed by
        score_candidates() (i.e. confidence/confidence_grade/score_breakdown
        keys must exist).
        """
        if not candidates:
            print("[GapScoring] No candidates to summarize.")
            return

        header = (
            f"{'START':>6} {'END':>6} "
            f"{'DIST':>7} {'DIR':>7} {'FACE':>7} "
            f"{'CONF':>7} {'GRADE':>10}"
        )
        separator = "-" * len(header)

        print("\n[GapScoring] Candidate confidence summary")
        print(separator)
        print(header)
        print(separator)

        for c in candidates:
            breakdown = c.get("score_breakdown", {})
            start = str(c.get("start_node", "?"))
            end = str(c.get("end_node", "?"))
            dist_raw = breakdown.get("distance_raw", 0.0)
            dir_raw = breakdown.get("direction_raw", 0.0)
            face_raw = breakdown.get("facing_raw", 0.0)
            confidence = c.get("confidence", 0.0)
            grade = c.get("confidence_grade", "?")

            print(
                f"{start:>6} {end:>6} "
                f"{dist_raw:7.3f} {dir_raw:7.3f} {face_raw:7.3f} "
                f"{confidence:7.3f} {grade:>10}"
            )

        print(separator)

        grade_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "REJECTED": 0}
        for c in candidates:
            grade_counts[c.get("confidence_grade", "REJECTED")] = (
                grade_counts.get(c.get("confidence_grade", "REJECTED"), 0) + 1
            )
        print(
            f"Totals -> HIGH: {grade_counts['HIGH']}  "
            f"MEDIUM: {grade_counts['MEDIUM']}  "
            f"LOW: {grade_counts['LOW']}  "
            f"REJECTED: {grade_counts['REJECTED']}"
        )
        print(separator + "\n")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def visualize(image, candidates, output_path):
        """
        Draw every candidate gap onto a copy of `image`, color coded by
        confidence grade:

            GREEN  = HIGH
            YELLOW = MEDIUM
            ORANGE = LOW
            GRAY   = REJECTED (dashed line)

        Node labels (start_node / end_node) are drawn next to each
        endpoint so the graph structure stays legible.

        Parameters
        ----------
        image : np.ndarray
            BGR image (e.g. the original road raster or skeleton overlay)
            to draw on. A copy is made; the original is not mutated.
        candidates : list[dict]
            Scored candidates (must already have confidence_grade,
            start_pos, end_pos keys).
        output_path : str
            File path the resulting PNG/JPG will be written to.

        Returns
        -------
        np.ndarray : the annotated image (also saved to output_path).
        """
        # Colors are BGR (OpenCV convention), not RGB.
        color_map = {
            "HIGH": (0, 200, 0),        # GREEN
            "MEDIUM": (0, 220, 220),    # YELLOW
            "LOW": (0, 140, 255),       # ORANGE
            "REJECTED": (150, 150, 150),  # GRAY
        }

        vis = image.copy()
        # Ensure the canvas is 3-channel BGR so colored lines are visible
        # even if a grayscale skeleton image was passed in.
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for candidate in candidates:
            start_pos = candidate.get("start_pos")
            end_pos = candidate.get("end_pos")
            if start_pos is None or end_pos is None:
                continue

            start_pt = (int(round(start_pos[0])), int(round(start_pos[1])))
            end_pt = (int(round(end_pos[0])), int(round(end_pos[1])))

            grade = candidate.get("confidence_grade", "REJECTED")
            color = color_map.get(grade, color_map["REJECTED"])

            if grade == "REJECTED":
                GapScoring._draw_dashed_line(vis, start_pt, end_pt, color, thickness=1)
            else:
                thickness = 3 if grade == "HIGH" else 2
                cv2.line(vis, start_pt, end_pt, color, thickness, cv2.LINE_AA)

            # Small circles mark the endpoints themselves.
            cv2.circle(vis, start_pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, end_pt, 4, color, -1, cv2.LINE_AA)

            # Node labels, offset slightly so they don't sit on top of
            # the endpoint marker.
            start_label = str(candidate.get("start_node", ""))
            end_label = str(candidate.get("end_node", ""))
            if start_label:
                cv2.putText(
                    vis, start_label,
                    (start_pt[0] + 6, start_pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
            if end_label:
                cv2.putText(
                    vis, end_label,
                    (end_pt[0] + 6, end_pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

            # Confidence value printed at the midpoint of the gap.
            mid_pt = (
                int(round((start_pt[0] + end_pt[0]) / 2)),
                int(round((start_pt[1] + end_pt[1]) / 2)),
            )
            confidence_text = f"{candidate.get('confidence', 0.0):.2f}"
            cv2.putText(
                vis, confidence_text, mid_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )

        # Legend box in the top-left corner.
        GapScoring._draw_legend(vis, color_map)

        cv2.imwrite(output_path, vis)
        print(f"[GapScoring] Visualization saved to: {output_path}")
        return vis

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=6):
        """Draw a dashed line between pt1 and pt2 (used for REJECTED gaps)."""
        pt1 = np.array(pt1, dtype=np.float64)
        pt2 = np.array(pt2, dtype=np.float64)
        total_length = np.linalg.norm(pt2 - pt1)
        if total_length == 0:
            return

        direction = (pt2 - pt1) / total_length
        num_dashes = max(int(total_length // dash_length), 1)

        for i in range(0, num_dashes, 2):
            seg_start = pt1 + direction * (i * dash_length)
            seg_end_len = min((i + 1) * dash_length, total_length)
            seg_end = pt1 + direction * seg_end_len
            cv2.line(
                img,
                (int(round(seg_start[0])), int(round(seg_start[1]))),
                (int(round(seg_end[0])), int(round(seg_end[1]))),
                color, thickness, cv2.LINE_AA
            )

    @staticmethod
    def _draw_legend(img, color_map):
        """Draw a small color legend box in the top-left corner of img."""
        x0, y0 = 10, 10
        row_height = 20
        box_width = 130
        box_height = row_height * len(color_map) + 10

        overlay = img.copy()
        cv2.rectangle(
            overlay, (x0, y0), (x0 + box_width, y0 + box_height),
            (0, 0, 0), -1
        )
        # Blend the black box in semi-transparently so it doesn't fully
        # obscure the underlying image.
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

        order = ["HIGH", "MEDIUM", "LOW", "REJECTED"]
        for idx, grade in enumerate(order):
            color = color_map[grade]
            y = y0 + 15 + idx * row_height
            cv2.line(img, (x0 + 8, y), (x0 + 28, y), color, 3, cv2.LINE_AA)
            cv2.putText(
                img, grade, (x0 + 34, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )