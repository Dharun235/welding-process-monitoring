"""Weldpool localization utilities.

The main strategy projects a wire-center axis and scans downward for the first
valid edge response near that axis.
"""

import cv2
import numpy as np


def _scan_profile(edge_image, point, direction, half_width_px):
    """
    Sample a short 1D profile through ``point`` along ``direction`` and return the
    closest edge pixel to the center together with its contiguous run width.
    """
    if half_width_px <= 0:
        return None, 0

    h, w = edge_image.shape[:2]
    center = np.asarray(point, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None, 0
    direction = direction / norm

    samples = []
    values = []
    for offset in range(-int(half_width_px), int(half_width_px) + 1):
        sample = center + direction * float(offset)
        x = int(round(float(sample[0])))
        y = int(round(float(sample[1])))
        samples.append((x, y))
        if 0 <= x < w and 0 <= y < h and edge_image[y, x] > 0:
            values.append(1)
        else:
            values.append(0)

    center_idx = int(half_width_px)
    hit_idx = None
    best_distance = None
    for idx, value in enumerate(values):
        if not value:
            continue
        distance = abs(idx - center_idx)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            hit_idx = idx

    if hit_idx is None:
        return None, 0

    left = hit_idx
    while left - 1 >= 0 and values[left - 1]:
        left -= 1
    right = hit_idx
    while right + 1 < len(values) and values[right + 1]:
        right += 1

    run_width = right - left + 1
    return samples[hit_idx], run_width


def _remove_near_circular_components(
    edge_mask,
    min_area_px,
    circularity_threshold,
):
    """
    Remove connected edge components that look like droplets (near-circular blobs).

    Circularity is computed as 4*pi*A/P^2 where 1.0 is a perfect circle.
    """
    if edge_mask is None or edge_mask.ndim != 2:
        return edge_mask

    cleaned = edge_mask.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < float(min_area_px):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity >= float(circularity_threshold):
            cv2.drawContours(cleaned, [contour], -1, 0, thickness=-1)

    return cleaned


def localize_weldpool(
    wire_edges,
    canny_edges,
    wiretip_location,
    basemetal_y,
    min_edge_width_px,
    scan_half_width_px,
    remove_circular_droplets,
    droplet_min_area_px,
    droplet_circularity_threshold,
    show_verification,
):
    """
    Approximate wire centerline from wire edges, extend it beyond the wire, then
    scan downward along that centerline until an edge pixel is found.

    Parameters:
    - wire_edges: Tuple/list with two line segments: ((x1,y1,x2,y2), (x1,y1,x2,y2))
    - canny_edges: Binary Canny edge image (H, W), non-zero means edge
    - wiretip_location: Wiretip point (x, y). Search starts at y = wiretip_y + 1.
    - basemetal_y: Optional maximum weldpool y-location in image coordinates.
      Candidates must satisfy y <= basemetal_y.
    - min_edge_width_px: Minimum contiguous edge width (in pixels) required to
      accept a detected weldpool edge point.
    - scan_half_width_px: Half-width of the perpendicular scan band around the
      projected wire centerline.
    - remove_circular_droplets: Remove near-circular components before weldpool scan.
    - droplet_min_area_px: Ignore very small contour fragments when circularity filtering.
    - droplet_circularity_threshold: Circularity threshold for filtering droplets.
    - show_verification: If True, show intermediate debug windows.

    Returns:
    - (x, y) of first edge hit when scanning downward, or None if not found.
    """

    if canny_edges is None or canny_edges.ndim != 2:
        return None

    if wire_edges is None or len(wire_edges) != 2:
        return None

    if wiretip_location is None:
        return None
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    edge_mask = (canny_edges > 0).astype(np.uint8)
    original_edge_mask = edge_mask.copy()
    removed_mask = np.zeros_like(edge_mask, dtype=np.uint8)
    if np.any(edge_mask):
        if remove_circular_droplets:
            edge_mask = _remove_near_circular_components(
                edge_mask,
                min_area_px=droplet_min_area_px,
                circularity_threshold=droplet_circularity_threshold,
            )
            removed_mask = ((original_edge_mask > 0) & (edge_mask == 0)).astype(np.uint8) * 255

        min_component_area = max(4, int(min_edge_width_px))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_component_area
        keep[0] = False
        if num_labels > 1 and not np.all(keep[1:]):
            filtered_edges = np.zeros_like(canny_edges)
            filtered_edges[keep[labels]] = 255
        else:
            filtered_edges = (edge_mask * 255).astype(canny_edges.dtype)
    else:
        filtered_edges = canny_edges

    dirs = []
    mids = []
    for line in wire_edges:
        x1, y1, x2, y2 = [float(v) for v in line]
        v = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        v = v / n
        # Prefer downward orientation in image coordinates.
        if v[1] < 0:
            v = -v
        dirs.append(v)
        mids.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))

    if not dirs:
        return None

    axis = np.mean(np.vstack(dirs), axis=0)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-6:
        return None
    axis = axis / axis_n
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = np.mean(np.vstack(mids), axis=0)

    h, w = filtered_edges.shape[:2]

    # Determine wire span along axis so the line can be extended beyond it.
    proj_vals = []
    for line in wire_edges:
        x1, y1, x2, y2 = [float(v) for v in line]
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        proj_vals.extend([(p1 - center) @ axis, (p2 - center) @ axis])

    s_min = float(np.min(proj_vals))
    s_max = float(np.max(proj_vals)) + 1000
    start_y = int(round(float(wiretip_location[1]))) + 3
    if basemetal_y is None:
        max_search_y = None
    else:
        max_search_y = int(np.clip(round(float(basemetal_y)), 0, h - 1))

    hit = None
    min_run_width = max(1, int(min_edge_width_px))
    scan_half_width_px = max(1, int(scan_half_width_px))

    # Sample from top (small s) to bottom (large s); accept the first edge found
    # within a narrow band around the projected wire axis.
    for s in range(int(np.floor(s_min)), int(np.ceil(s_max)) + 1):
        p = center + axis * float(s)
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if y < start_y:
            continue
        if max_search_y is not None and y > max_search_y:
            break
        candidate, run_width = _scan_profile(filtered_edges, p, normal, scan_half_width_px)
        if (
            candidate is not None
            and run_width >= min_run_width
            and (max_search_y is None or candidate[1] <= max_search_y)
        ):
            hit = candidate
            break

    if hit is not None and max_search_y is not None and int(hit[1]) > max_search_y:
        hit = None

    if show_verification:
        debug_window_name = "Weldpool"
        original_vis = (original_edge_mask * 255).astype(np.uint8)
        filtered_vis = filtered_edges.astype(np.uint8)
        overlay = cv2.cvtColor(filtered_vis, cv2.COLOR_GRAY2BGR)

        if np.any(removed_mask):
            overlay[removed_mask > 0] = (0, 0, 255)

        for line in wire_edges:
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

        wx, wy = int(round(float(wiretip_location[0]))), int(round(float(wiretip_location[1])))
        cv2.circle(overlay, (wx, wy), 3, (255, 255, 0), -1)

        center_start = (int(round(float(wiretip_location[0] - normal[0] * scan_half_width_px))),
                        int(round(float(wiretip_location[1] - normal[1] * scan_half_width_px))))
        center_end = (int(round(float(wiretip_location[0] + normal[0] * scan_half_width_px))),
                      int(round(float(wiretip_location[1] + normal[1] * scan_half_width_px))))
        cv2.line(overlay, center_start, center_end, (0, 255, 255), 1)

        axis_start_s = max(s_min, float(start_y - center[1]) / float(axis[1])) if abs(float(axis[1])) > 1e-6 else s_min
        axis_start = center + axis * float(axis_start_s)
        if max_search_y is not None and abs(float(axis[1])) > 1e-6:
            axis_end_s = min(s_max, float(max_search_y - center[1]) / float(axis[1]))
        else:
            axis_end_s = s_max
        axis_end = center + axis * float(axis_end_s)
        cv2.line(
            overlay,
            (int(round(float(axis_start[0]))), int(round(float(axis_start[1])))),
            (int(round(float(axis_end[0]))), int(round(float(axis_end[1])))),
            (255, 255, 0),
            1,
        )

        #if max_search_y is not None:
        #    cv2.line(overlay, (0, max_search_y), (overlay.shape[1], max_search_y), (0, 165, 255), 1)

        if hit is not None:
            cv2.circle(overlay, hit, 5, (255, 0, 255), -1)

        cv2.imwrite("weldpool_w_droplets.png", overlay)

        #cv2.imshow(f"{debug_window_name} - Canny", original_vis)
        #cv2.imshow(f"{debug_window_name} - Filtered", filtered_vis)
        #cv2.imshow(f"{debug_window_name} - Overlay", overlay)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return hit
