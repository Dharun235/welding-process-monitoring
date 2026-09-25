"""Tapering-point localization from tracked wire-edge continuation."""

import cv2
import numpy as np


def _fit_line_params(line):
    """Fit a line model and force a downward-positive direction."""
    pts = np.array(
        [[line[0], line[1]], [line[2], line[3]]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(np.asarray(vx).reshape(-1)[0])
    vy = float(np.asarray(vy).reshape(-1)[0])
    x0 = float(np.asarray(x0).reshape(-1)[0])
    y0 = float(np.asarray(y0).reshape(-1)[0])
    if vy < 0:
        vx = -vx
        vy = -vy
    return vx, vy, x0, y0


def _line_x_at_y(fit_params, y):
    """Evaluate fitted line x-coordinate at a given y position."""
    vx, vy, x0, y0 = fit_params
    if abs(vy) < 1e-6:
        return x0
    t = (y - y0) / vy
    return x0 + t * vx


def _search_taper_stop(edges, line, fit_params, search_window_radius, vicinity_radius, max_consecutive_misses):
    """Track edge support downward until taper-stop conditions are met."""
    height, width = edges.shape[:2]

    top_y = min(line[1], line[3])
    bottom_y = max(line[1], line[3])
    start_y = int(round(bottom_y - 0.2 * (bottom_y - top_y)))
    last_valid = None
    support_points = []
    miss_count = 0
    stop_reason = "image_bottom"

    for y in range(max(start_y, 0), height):
        x_expected = _line_x_at_y(fit_params, y)
        if x_expected < -search_window_radius or x_expected >= width + search_window_radius:
            stop_reason = "left_frame"
            break

        x_center = int(round(x_expected))
        x_min = max(0, x_center - search_window_radius)
        x_max = min(width - 1, x_center + search_window_radius)
        row_hits = np.flatnonzero(edges[y, x_min:x_max + 1] > 0)

        if row_hits.size == 0:
            miss_count += 1
            if miss_count >= max_consecutive_misses:
                stop_reason = "edge_missing"
                break
            continue

        hit_x = row_hits + x_min
        nearest_idx = int(np.argmin(np.abs(hit_x - x_expected)))
        nearest_x = int(hit_x[nearest_idx])
        distance = abs(nearest_x - x_expected)

        if distance > vicinity_radius:
            stop_reason = "outside_vicinity"
            break

        miss_count = 0
        last_valid = (nearest_x, y)
        support_points.append(last_valid)

    return last_valid, support_points, stop_reason


def localize_tapering_point(edges, wire_edges, search_window_radius, vicinity_radius, max_consecutive_misses):
    """
    Track both detected wire edges downward from the midpoint of each segment
    along their fitted lines and return the midpoint between the two stop points.

    Parameters:
    - edges: Binary edge image (e.g., from Canny).
    - wire_edges: List of two lines [(x1, y1, x2, y2), (x1, y1, x2, y2)] representing the detected wire edges.
    - search_window_radius: Horizontal radius around the expected line position to search for edge pixels.
    - vicinity_radius: Maximum allowed distance from the expected line position for a valid edge pixel.
    - max_consecutive_misses: Maximum number of consecutive rows without valid edge detections before stopping the search.

    Returns:
    - (x, y) tapering point, or None if no valid stop point is found.
    """
    if wire_edges is None or len(wire_edges) != 2:
        return None

    line_a, line_b = wire_edges
    fit_a = _fit_line_params(line_a)
    fit_b = _fit_line_params(line_b)

    stop_a, support_a, reason_a = _search_taper_stop(edges, line_a, fit_a, search_window_radius, vicinity_radius, max_consecutive_misses)
    stop_b, support_b, reason_b = _search_taper_stop(edges, line_b, fit_b, search_window_radius, vicinity_radius, max_consecutive_misses)

    if stop_a is None and stop_b is None:
        return None

    if stop_a is not None and stop_b is not None:
        tapering_point = (
            int(round(0.5 * (stop_a[0] + stop_b[0]))),
            int(round(0.5 * (stop_a[1] + stop_b[1]))),
        )
    else:
        tapering_point = stop_a if stop_a is not None else stop_b

    vis = edges.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def draw_debug(image, line, fit_params, support_points, stop_point, color):
        height, width = image.shape[:2]
        x_top = int(round(_line_x_at_y(fit_params, 0)))
        x_bottom = int(round(_line_x_at_y(fit_params, height - 1)))
        pt1 = (int(np.clip(x_top, 0, width - 1)), 0)
        pt2 = (int(np.clip(x_bottom, 0, width - 1)), height - 1)

        cv2.line(image, pt1, pt2, color, 1)
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color, 2)
        for point in support_points:
            cv2.circle(image, point, 1, (255, 255, 0), -1)
        if stop_point is not None:
            cv2.circle(image, stop_point, 4, (0, 255, 255), -1)

    #draw_debug(vis, line_a, fit_a, support_a, stop_a, (0, 255, 0))
    #draw_debug(vis, line_b, fit_b, support_b, stop_b, (0, 0, 255))
    #cv2.circle(vis, tapering_point, 5, (255, 0, 255), -1)
    #cv2.putText(vis, f"L1: {reason_a}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #cv2.putText(vis, f"L2: {reason_b}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #cv2.imshow("Tapering Point Search", vis)
    #cv2.waitKey(0)
    #cv2.destroyWindow("Tapering Point Search")

    return tapering_point
