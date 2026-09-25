"""Search-space definition around the wire for robust local analysis."""

import cv2
import numpy as np


def _order_box_points(box):
    """Order quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _normalize_line_direction(line):
    """Normalize line endpoint order to keep direction stable across frames."""
    x1, y1, x2, y2 = [float(v) for v in line]
    # Prefer a downward-pointing direction for near-vertical wire edges.
    if (y2 < y1) or (y2 == y1 and x2 < x1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2


def define_searchspace(
    img,
    wire_edges,
    weldpool_y,
    tooltip,
    top_shift_px,
    length_scale,
    width_scale,
):
    """
    Define a search space around the wire by fitting a rectangle to the detected wire edges.
    If the tooltip is provided the top of the rectangle will be defined by the tooltip location.
    If weldpool_y is provided, the bottom of the rectangle will be defined by the the weld pool location.
    Parameters:
    - img: Preprocessed grayscale image (numpy array).
    - wire_edges: touple of two lines representing the detected wire edges, each line is (x1, y1, x2, y2).
    - weldpool_y: y-coordinate of the localized weld pool (int) or None if not localized.
    - tooltip: touple of two points representing the detected wire tool transition, each point is (x, y) or None if not detected.
    - top_shift_px: number of pixels to shift the search space top downwards from the tooltip
    - length_scale: scale factor to apply to the search space length (float)
    - width_scale: scale factor to apply to the search space width (float)

    Returns:
    - cropped: Cropped image of the defined search space (numpy array).
    - M_inv: Inverse perspective transform matrix to map points from the search space back to the original image (numpy array).
    - box_ord: Coordinates of the search space corners in the original image (numpy array of shape (4, 2)).
    """
    if img is None or wire_edges is None or len(wire_edges) != 2:
        return None

    if length_scale <= 0 or width_scale <= 0:
        return None

    try:
        l1, l2 = wire_edges
        x1, y1, x2, y2 = _normalize_line_direction(l1)
        x3, y3, x4, y4 = _normalize_line_direction(l2)
    except (TypeError, ValueError):
        return None

    # Identify longer line
    len1 = float(np.hypot(x2 - x1, y2 - y1))
    len2 = float(np.hypot(x4 - x3, y4 - y3))
    if len1 < 1e-6 and len2 < 1e-6:
        return None

    if len2 > len1:
        (x1, y1, x2, y2), (x3, y3, x4, y4) = (x3, y3, x4, y4), (x1, y1, x2, y2)
        len1 = len2

    dx = x2 - x1
    dy = y2 - y1
    inv_len = 1.0 / max(len1, 1e-6)
    dir_len = np.array([dx, dy], dtype=np.float32) * inv_len

    # Align the second line direction with the reference line to keep projections stable.
    dir2 = np.array([x4 - x3, y4 - y3], dtype=np.float32)
    if np.dot(dir2, dir_len) < 0:
        x3, y3, x4, y4 = x4, y4, x3, y3

    nx = -dy * inv_len
    ny = dx * inv_len

    d3 = (x3 - x1) * nx + (y3 - y1) * ny
    d4 = (x4 - x1) * nx + (y4 - y1) * ny
    d = (d3 + d4) / 2.0
    if abs(d) < 1e-6:
        return None

    p1 = np.array([x1, y1], dtype=np.float32)
    p2 = np.array([x2, y2], dtype=np.float32)
    offset = np.array([nx * d, ny * d], dtype=np.float32)
    p3 = p2 + offset
    p4 = p1 + offset

    center = (p1 + p2 + p3 + p4) / 4.0
    dir_wid = np.array([nx, ny], dtype=np.float32)

    half_len = (len1 * length_scale) / 2.0
    width = max(abs(d) * width_scale, 1.0)
    half_wid = width / 2.0

    # Searchspace top selection
    if tooltip is not None:
        top_center = None
        try:
            if len(tooltip) >= 2:
                tp0 = np.array(tooltip[0], dtype=np.float32)
                tp1 = np.array(tooltip[1], dtype=np.float32)
                top_center = (tp0 + tp1) / 2.0
            elif len(tooltip) == 1:
                tp0 = np.array(tooltip[0], dtype=np.float32)
                # For a single point tooltip, only trust its y-position and keep current x-center.
                top_center = np.array([center[0], tp0[1]], dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            top_center = None

        if top_center is not None and np.isfinite(top_center).all():
            top_center = top_center + dir_len * float(top_shift_px)
            center = top_center + dir_len * half_len

    p1 = center - dir_len * half_len - dir_wid * half_wid
    p2 = center + dir_len * half_len - dir_wid * half_wid
    p3 = center + dir_len * half_len + dir_wid * half_wid
    p4 = center - dir_len * half_len + dir_wid * half_wid

    # Searchspace bottom selection based on weld pool
    if weldpool_y is not None and abs(float(dir_len[1])) > 1e-6:
        bottom_center = (p2 + p3) / 2.0
        t = (float(weldpool_y) - float(bottom_center[1])) / float(dir_len[1])
        p2 = p2 + dir_len * t
        p3 = p3 + dir_len * t

    box = np.array([p1, p2, p3, p4], dtype=np.float32)
    if not np.isfinite(box).all():
        return None

    box_ord = _order_box_points(box)

    w1 = np.linalg.norm(box_ord[1] - box_ord[0])
    w2 = np.linalg.norm(box_ord[2] - box_ord[3])
    h1 = np.linalg.norm(box_ord[3] - box_ord[0])
    h2 = np.linalg.norm(box_ord[2] - box_ord[1])
    width = int(round(max(w1, w2)))
    height = int(round(max(h1, h2)))
    max_dim = int(max(img.shape[0], img.shape[1]) * 4)

    if width <= 1 or height <= 1 or width > max_dim or height > max_dim:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    try:
        M = cv2.getPerspectiveTransform(box_ord, dst)
        cropped = cv2.warpPerspective(img, M, (width, height))
        M_inv = cv2.getPerspectiveTransform(dst, box_ord)
    except cv2.error:
        return None

    return (cropped, M_inv, box_ord.astype(np.intp))
