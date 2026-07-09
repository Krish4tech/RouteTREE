"""
graph_healer.py
================

RouteTREE Module 5: GraphHealer

Purpose
-------
Takes the NetworkX road graph `G` (nodes keyed by label strings such as
"A", "B", "C", ...) together with a list of gap candidates that have
already been scored by `gap_scoring.GapScoring.score_candidates()`, and
automatically reconnects roads that were broken by raster/skeletonization
artifacts.

A candidate is "healed" (turned into a real graph edge) only if:
    1. It was NOT hard-rejected upstream by CandidateMatcher
       (candidate["rejected"] is False), AND
    2. Its final confidence (from GapScoring) is >= confidence_threshold.

Candidates that fail either check are left out of the graph, and any
candidate whose endpoints are already connected by an existing edge is
skipped (we never overwrite or duplicate an existing edge).

Dependencies: Python standard library, NetworkX, OpenCV (cv2), NumPy only.
"""

import cv2
import numpy as np
import networkx as nx


class GraphHealer:
    """
    Automatically reconnects interrupted roads in a NetworkX road graph
    by inserting new edges for high-confidence gap candidates.
    """

    def __init__(self, confidence_threshold=0.50):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum GapScoring confidence (in [0, 1]) a candidate must
            have, in addition to not being upstream-rejected, before its
            gap is healed with a new edge.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be within [0, 1], got "
                f"{confidence_threshold}"
            )
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def heal_graph(self, G, candidates):
        """
        Main method. Walks every scored candidate and inserts a new edge
        into G for every candidate that passes the confidence threshold
        and is not already connected.

        Parameters
        ----------
        G : nx.Graph
            The road graph to heal. Modified in place (and also returned
            for convenience/chaining).
        candidates : list[dict]
            Candidates already processed by GapScoring.score_candidates(),
            i.e. each dict must contain at least:
                start_node, end_node, distance,
                confidence, confidence_grade, rejected

        Returns
        -------
        (G, summary) : tuple[nx.Graph, dict]
            G       : the same graph object, now containing healed edges.
            summary : {
                "healed": int,
                "skipped_low_confidence": int,
                "skipped_existing": int,
            }
        """
        summary = {
            "healed": 0,
            "skipped_low_confidence": 0,
            "skipped_existing": 0,
        }

        for candidate in candidates:
            start_node = candidate.get("start_node")
            end_node = candidate.get("end_node")

            # Malformed candidate (missing endpoint labels) -- nothing
            # sensible to do with it, so we don't touch the graph and we
            # don't count it in any bucket, but we do warn so it's not a
            # silent data-loss bug.
            if start_node is None or end_node is None:
                print(
                    "[GraphHealer] WARNING: candidate missing start_node/"
                    f"end_node, skipping entirely: {candidate}"
                )
                continue

            is_rejected = bool(candidate.get("rejected", False))
            confidence = float(candidate.get("confidence", 0.0))

            # Gate 1: upstream rejection OR below confidence threshold.
            # These are checked together because a candidate that was
            # hard-rejected always has confidence forced to 0.0 by
            # GapScoring anyway -- but we check `rejected` explicitly too,
            # in case candidates are ever fed in from a source that didn't
            # zero out confidence for rejected entries.
            if is_rejected or confidence < self.confidence_threshold:
                summary["skipped_low_confidence"] += 1
                continue

            # Gate 2: don't insert if the endpoints are already connected.
            # (Also guards against inserting a duplicate healed edge if
            # heal_graph() is accidentally called twice on the same G.)
            if G.has_edge(start_node, end_node):
                summary["skipped_existing"] += 1
                continue

            inserted = self.insert_edge(G, candidate)
            if inserted:
                summary["healed"] += 1
            else:
                # insert_edge() only returns False for malformed
                # candidates, which we already filtered above, so this
                # branch is a defensive fallback rather than an expected
                # path.
                summary["skipped_low_confidence"] += 1

        return G, summary

    # ------------------------------------------------------------------
    # Single edge insertion
    # ------------------------------------------------------------------

    def insert_edge(self, G, candidate):
        """
        Insert one healed edge into G for the given candidate.

        Adds nodes for start_node/end_node if they are somehow not
        already present in G (defensive -- in the normal pipeline both
        endpoints already exist as graph nodes from NetworkBuilder).

        Edge attributes set:
            weight     : float  -- Euclidean pixel distance of the gap
            confidence : float  -- GapScoring confidence
            healed     : True   -- always True for edges inserted here
            grade      : str    -- "HIGH" / "MEDIUM" / "LOW"

        Returns
        -------
        bool : True if the edge was inserted, False if the candidate was
               malformed (missing endpoint labels) and nothing was done.
        """
        start_node = candidate.get("start_node")
        end_node = candidate.get("end_node")
        if start_node is None or end_node is None:
            return False

        if not G.has_node(start_node):
            G.add_node(start_node)
        if not G.has_node(end_node):
            G.add_node(end_node)

        distance = float(candidate.get("distance", 0.0))
        confidence = float(candidate.get("confidence", 0.0))
        grade = candidate.get("confidence_grade", "LOW")

        G.add_edge(
            start_node,
            end_node,
            weight=distance,
            confidence=confidence,
            healed=True,
            grade=grade,
        )
        return True

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_healed_edges(self, image, G, positions, candidates, output_path):
        """
        Draw the healed graph state onto a copy of `image`:

            GREEN        = original (pre-existing) edges in G
            BLUE (thick) = healed edges (edges with healed=True in G)
            GRAY dashed  = rejected / below-threshold candidates that
                           were NOT inserted
            Node labels are drawn at every node that appears in an edge
            or a candidate, so the healed structure stays legible.
            A legend is drawn in the top-left corner.

        Parameters
        ----------
        image : np.ndarray
            BGR image (or grayscale, which will be converted) to draw on.
            A copy is made; the original is not mutated.
        G : nx.Graph
            The (already healed) road graph.
        positions : dict[str, tuple]
            Maps node label -> (x, y) pixel coordinate.
        candidates : list[dict]
            The full scored candidate list (same list passed to
            heal_graph), used to draw rejected/skipped gaps in gray.
        output_path : str
            File path the resulting PNG/JPG will be written to.

        Returns
        -------
        np.ndarray : the annotated image (also saved to output_path).
        """
        color_original = (0, 200, 0)      # GREEN
        color_healed = (255, 120, 0)      # BLUE (OpenCV BGR)
        color_rejected = (150, 150, 150)  # GRAY

        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # --- 1. Draw every edge currently in G -------------------------
        # Edges are distinguished purely by the "healed" attribute that
        # GraphHealer stamps onto edges it inserts. Any edge lacking that
        # attribute (or with healed=False) is treated as an original,
        # pre-existing road segment.
        for u, v, data in G.edges(data=True):
            if u not in positions or v not in positions:
                continue

            pt1 = self._to_point(positions[u])
            pt2 = self._to_point(positions[v])

            if data.get("healed", False):
                cv2.line(vis, pt1, pt2, color_healed, 4, cv2.LINE_AA)
            else:
                cv2.line(vis, pt1, pt2, color_original, 2, cv2.LINE_AA)

        # --- 2. Draw rejected / skipped candidates as dashed gray ------
        # We recompute the accept/reject decision here (rather than
        # trusting edge presence in G) so a candidate that was skipped
        # for being below-threshold is still shown, even though its
        # endpoints might coincidentally already share an original edge
        # via some other path.
        for candidate in candidates:
            start_node = candidate.get("start_node")
            end_node = candidate.get("end_node")
            if start_node not in positions or end_node not in positions:
                continue

            is_rejected = bool(candidate.get("rejected", False))
            confidence = float(candidate.get("confidence", 0.0))
            was_healed = confidence >= self.confidence_threshold and not is_rejected

            if was_healed:
                # Already drawn (in blue) from the G.edges() pass above.
                continue

            pt1 = self._to_point(positions[start_node])
            pt2 = self._to_point(positions[end_node])
            self._draw_dashed_line(vis, pt1, pt2, color_rejected, thickness=1)

        # --- 3. Draw node labels ----------------------------------------
        # Only label nodes that are actually referenced by an edge or a
        # candidate, so the image doesn't get cluttered with every
        # isolated skeleton point in the graph.
        labeled_nodes = set()
        for u, v in G.edges():
            labeled_nodes.add(u)
            labeled_nodes.add(v)
        for candidate in candidates:
            labeled_nodes.add(candidate.get("start_node"))
            labeled_nodes.add(candidate.get("end_node"))
        labeled_nodes.discard(None)

        for node in labeled_nodes:
            if node not in positions:
                continue
            pt = self._to_point(positions[node])
            cv2.circle(vis, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                vis, str(node), (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

        # --- 4. Legend ---------------------------------------------------
        self._draw_legend(vis, color_original, color_healed, color_rejected)

        cv2.imwrite(output_path, vis)
        print(f"[GraphHealer] Visualization saved to: {output_path}")
        return vis

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_point(pos):
        """Convert a (x, y) float/np coordinate pair to an int (x, y) tuple."""
        return (int(round(pos[0])), int(round(pos[1])))

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=6):
        """Draw a dashed line between pt1 and pt2 (used for rejected gaps)."""
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
    def _draw_legend(img, color_original, color_healed, color_rejected):
        """Draw a small color legend box in the top-left corner of img."""
        x0, y0 = 10, 10
        row_height = 20
        box_width = 170
        rows = [
            ("Original edge", color_original),
            ("Healed edge", color_healed),
            ("Rejected gap", color_rejected),
        ]
        box_height = row_height * len(rows) + 10

        overlay = img.copy()
        cv2.rectangle(
            overlay, (x0, y0), (x0 + box_width, y0 + box_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

        for idx, (label, color) in enumerate(rows):
            y = y0 + 15 + idx * row_height
            cv2.line(img, (x0 + 8, y), (x0 + 28, y), color, 3, cv2.LINE_AA)
            cv2.putText(
                img, label, (x0 + 34, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )