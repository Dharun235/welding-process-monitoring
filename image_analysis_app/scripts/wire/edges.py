"""Wire-edge candidate extraction using Hough segments merged into fitted lines."""

import cv2
import numpy as np


def _fit_line(points):
    """Fit a line and keep its direction pointing downward when possible."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(np.asarray(vx).reshape(-1)[0])
    vy = float(np.asarray(vy).reshape(-1)[0])
    x0 = float(np.asarray(x0).reshape(-1)[0])
    y0 = float(np.asarray(y0).reshape(-1)[0])
    if vy < 0:
        vx = -vx
        vy = -vy
    return vx, vy, x0, y0


def _project_points(points, fit_params):
    """Project 2D points onto the fitted line direction."""
    vx, vy, x0, y0 = fit_params
    pts = np.asarray(points, dtype=np.float32)
    return (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy


def _point_on_line(fit_params, t):
    """Evaluate a fitted line point at scalar position t."""
    vx, vy, x0, y0 = fit_params
    return np.array([x0 + t * vx, y0 + t * vy], dtype=np.float32)


def _normalize_segment_direction(line):
    """Return endpoints ordered from top to bottom for stable downstream use."""
    x1, y1, x2, y2 = [int(round(v)) for v in line]
    if (y2 < y1) or (y2 == y1 and x2 < x1):
        return x2, y2, x1, y1
    return x1, y1, x2, y2


def _build_segment_candidate(x1, y1, x2, y2):
    """Build a normalized segment descriptor."""
    x1, y1, x2, y2 = _normalize_segment_direction((x1, y1, x2, y2))
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    angle = float(np.arctan2(dy, dx)) if float(np.arctan2(dy, dx)) >= 0 else float(np.arctan2(dy, dx)) + np.pi
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    return {
        "line": (x1, y1, x2, y2),
        "points": points,
        "length": length,
        "angle": angle,
        "avg_x": 0.5 * (x1 + x2),
        "y_min": min(y1, y2),
        "y_max": max(y1, y2),
    }


def _projection_gap(interval_a, interval_b):
    """Distance between projected intervals along a line."""
    a_min, a_max = sorted(interval_a)
    b_min, b_max = sorted(interval_b)
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def _segments_can_merge(seg_a, seg_b, merge_angle_tol, merge_gap_px, merge_line_distance_px):
    """Check whether two raw Hough segments likely belong to the same edge."""
    if abs(seg_a["angle"] - seg_b["angle"]) > merge_angle_tol:
        return False

    fit_params = _fit_line(np.vstack((seg_a["points"], seg_b["points"])))
    vx, vy, x0, y0 = fit_params
    nx = -vy
    ny = vx
    all_points = np.vstack((seg_a["points"], seg_b["points"]))
    distances = np.abs((all_points[:, 0] - x0) * nx + (all_points[:, 1] - y0) * ny)
    if float(np.max(distances)) > merge_line_distance_px:
        return False

    proj_a = _project_points(seg_a["points"], fit_params)
    proj_b = _project_points(seg_b["points"], fit_params)
    gap = _projection_gap((float(np.min(proj_a)), float(np.max(proj_a))), (float(np.min(proj_b)), float(np.max(proj_b))))
    return gap <= merge_gap_px


def _merge_segments_into_lines(candidates, min_length, merge_angle_tol, merge_gap_px, merge_line_distance_px):
    """Merge compatible raw segments and convert each group into one fitted line."""
    if not candidates:
        return []

    neighbors = [set() for _ in candidates]
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if _segments_can_merge(candidates[i], candidates[j], merge_angle_tol, merge_gap_px, merge_line_distance_px):
                neighbors[i].add(j)
                neighbors[j].add(i)

    groups = []
    visited = set()
    for start in range(len(candidates)):
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nxt in neighbors[idx]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        groups.append(component)

    merged = []
    for group in groups:
        group_points = np.vstack([candidates[idx]["points"] for idx in group])
        fit_params = _fit_line(group_points)
        t_values = _project_points(group_points, fit_params)
        line_start = _point_on_line(fit_params, float(np.min(t_values)))
        line_end = _point_on_line(fit_params, float(np.max(t_values)))
        line = _normalize_segment_direction((line_start[0], line_start[1], line_end[0], line_end[1]))
        merged_candidate = _build_segment_candidate(*line)
        if merged_candidate["length"] < min_length:
            continue
        merged_candidate["support_count"] = len(group)
        merged.append(merged_candidate)

    return merged


def identify_wire_edges(
    img,edges,
    basemetal_y,
    hough_threshold,
    min_line_length,
    max_line_gap,
    rho,
    theta,
    min_length,
    angle_tolerance_deg,
    vertical_tolerance_deg,
    min_x_distance,
    max_x_distance,
):
    '''
    Find two wire-edge lines by detecting short Hough segments, merging segments
    that are likely part of the same physical edge, and fitting one longer line
    to each merged group using the observed support extent.
    Parameters:
    - edges: Canny edges of a preprocessed image (numpy array).
    - basemetal_y: Basemetal y-location; valid line endpoints must satisfy y < basemetal_y.
    - hough_threshold: Threshold for the Hough line transform.
    - min_line_length: Target minimum line length for raw Hough proposals.
    - max_line_gap: Maximum allowed gap between line segments to treat them as a single line for the Hough line transform.
    - rho: Distance resolution in pixels of the Hough space.
    - min_length: Minimum length of each merged line candidate in pixels.
    - angle_tolerance_deg: Maximum allowed angle difference between the two line candidates in degrees.
    - vertical_tolerance_deg: Maximum allowed deviation from vertical in degrees for each raw segment.
    - min_x_distance: Minimum allowed distance in x between the average x of the two line candidates.
    - max_x_distance: Maximum allowed distance in x between the average x of the two line candidates.

    Returns:
    - A tuple of two line segments: ((x1,y1,x2,y2), (x1,y1,x2,y2)) or None if no suitable pair is found.
    '''

    raw_min_length = max(8.0, 0.4 * float(min_length))
    hough_min_line_length = max(8, int(round(min(min_line_length, raw_min_length))))

    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        print("No wire edges detected")
        return None

    vertical_center = 0.5 * np.pi
    vertical_tol = np.deg2rad(vertical_tolerance_deg)
    merge_angle_tol = np.deg2rad(max(angle_tolerance_deg, 3.0))
    merge_gap_px = max_line_gap
    merge_line_distance_px = float(max(4, round(0.15 * min_x_distance)))

    raw_candidates = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if basemetal_y is not None and (y1 >= basemetal_y or y2 >= basemetal_y):
            continue

        candidate = _build_segment_candidate(x1, y1, x2, y2)
        if candidate["length"] < raw_min_length:
            continue

        if abs(candidate["angle"] - vertical_center) > vertical_tol:
            continue

        raw_candidates.append(candidate)

    if len(raw_candidates) < 2:
        return None

    candidates = _merge_segments_into_lines(
        raw_candidates,
        min_length=min_length,
        merge_angle_tol=merge_angle_tol,
        merge_gap_px=merge_gap_px,
        merge_line_distance_px=merge_line_distance_px,
    )

    if len(candidates) < 2:
        return None

    tol = np.deg2rad(angle_tolerance_deg)
    best_pair = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if abs(candidates[i]["angle"] - candidates[j]["angle"]) > tol:
                continue

            x1a, y1a, x2a, y2a = candidates[i]["line"]
            x1b, y1b, x2b, y2b = candidates[j]["line"]

            y_min_a, y_max_a = min(y1a, y2a), max(y1a, y2a)
            y_min_b, y_max_b = min(y1b, y2b), max(y1b, y2b)
            if y_max_a < y_min_b or y_max_b < y_min_a:
                continue

            avg_x_a = candidates[i]["avg_x"]
            avg_x_b = candidates[j]["avg_x"]
            x_distance = abs(avg_x_a - avg_x_b)
            if x_distance < min_x_distance or x_distance > max_x_distance:
                continue

            score = candidates[i]["length"] + candidates[j]["length"]
            if score > best_score:
                best_score = score
                best_pair = (candidates[i], candidates[j])

    if best_pair is None:
        print("No suitable pair of wire edges found")
        return None

    ordered_pair = sorted(best_pair, key=lambda candidate: candidate["avg_x"])
    return tuple(candidate["line"] for candidate in ordered_pair)
